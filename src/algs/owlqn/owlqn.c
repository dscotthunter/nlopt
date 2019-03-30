#include <stdlib.h> 
#include <math.h>
#include <stdio.h>

#include "owlqn.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int owlqn_verbose = 1;

typedef struct { 
    nlopt_func f; /* This is just the loss function */
    void *f_data;
    int m; /* the number of variables to remember */
    double *lambda, *gradtmp;
    nlopt_stopping *stop;
} owlqn_data;

/* This is used to store information from each step, based on the paper
 * Updating Quasi-Newton Matrices with Limited Storage
 * Jorge Nocedal */
typedef struct {
    double alpha;
    double *s;
    double *y;
    double ys; 
} iteration_data;


/* Simple helper functions for working with vectors */
/* computes l1norm of a vector */
double l1norm_vector(int n, double *x){
    int i;
    double norm = 0.;
    for (i = 0; i < n; ++i){
        norm += fabs(x[i]);
    }
    return norm;
}

/* computes dot product of x and y */
void vecdot_owlqn(double *out, int n, double *x, double *y){
    int i;
    double *out = 0.;
    for (i = 0; i < n; ++i){
        *out += x[i] * y[i];
    }
    return;
}

/* computes x = x + c y */
void vecadd_owlqn(double *x, double *y, double *c, int n){
    int i;
    for (i = 0; i < n; ++i){
        x[i] += (*c) * y[i];
    }
    return;
}

/* Changes x to be y - z */ 
void vecdiff_owlqn(double *x, double *y, double *z, int n){
    int i;

    for (i = 0; i < n; ++i){
        x[i] = y[i] - z[i];
    }
    return;
}

/* returns x, where x is projected onto orthant containing y */
void owlqn_project(double *x, double *y, int n){
    int i;

    for (i=0; i < n; ++i){
        if (x[i] * y[i] <= 0){
            x[i] = 0;
        }
    }
    return;
}


/* OWL-QN specific helper functions */
void pseudo_gradient(double* pseudograd,
        const double *x,
        const double *grad, 
        const int n,
        const double lambda,
        double *gmax
        )
{
    
    int i;    
    for (i = 0; i < n; ++i){
        if (x[i] < 0.){
            pseudograd[i] = grad[i] - lambda;
        }
        else if (x[i] > 0.){
            pseudograd[i] = grad[i] + lambda;
        }
        else{
            /* Not differentiable at this point */
            if (grad[i] < - lambda){
                /* Then right partial is negative so we take that */
                pseudograd[i] = grad[i] + lambda;
            }
            else if (lambda < grad[i]){
                /* Then left partial derivative is positive so take that */
                pseudograd[i] = grad[i] - lambda;
            }
            else{
                /* otherwise case in paper */
                pseudograd[i] = 0.;
            }
        }

        if (i<=0){
            *gmax = fabs(pseudograd[i]);
        }
        else if(*gmax >= fabs(pseudograd[i])){
            *gamx = fabs(pseudograd[i]);
        }
    }
    return;
}





void check_stopping_criteria(int *n, int *owlqn_iters, 
        double *xcur, double *xprev,
        double *fcur, double *fprev, 
        double *pseudo_gradient, nlopt_result *ret, 
        nlopt_stopping *stop)
{
  if (nlopt_stop_forced(stop)) {
    *ret = NLOPT_FORCED_STOP;
    return;
  }
  if (*f <= stop->minf_max){
    *ret = NLOPT_MINF_MAX_REACHED;
    return;
  }
  if (*gmax <= 1e-8){
    *ret = NLOPT_SUCCESS;
    return;
  }
  if (nlopt_stop_ftol(stop, *fcur, *fprev)){
    *ret = NLOPT_FTOL_REACHED;
    return;
  }
  if (nlopt_stop_x(stop, xcur, xprev) && (*owlqn_iters)){
    *ret = NLOPT_XTOL_REACHED;
    return;
  }
  if (nlopt_stop_evals(stop)){
    *ret = NLOPT_MAXEVAL_REACHED;
  }
  return;
}






int line_search_owlqn(int n, 
        double *xcur, double *fcur, 
        double *cgrad, double *direction,
        double *step, double *orthant, 
        const double *xprev, const double *pseudograd_prev, 
        owlqn_data *d, nlopt_stopping *stop)
{
    int i, count =0;
    double width = 0.5, normx = 0.;
    double f_initial = *fcur, dgtest;
    double gamma = 1e-4;
    double min_step = 1e-20;
    double max_step = 1e20;
    int max_linesearch = 20;

    if (*step <= 0.){
        return -1;
    }

    /* Choose the orthant search direction */
    for (i = 0; i < n; ++i){
        orthant[i] = (xprev[i] == 0.) ? -pseudograd_prev[i] : pseudograd_prev[i];
    }


    for (;;){
        /* Update the current point */
        memcpy(xcur, xprev, sizeof(double) * n);
        vecadd_owlqn(xcur, direction, step, n);

        /* Project this new point onto the correct orthant */
        owlqn_project(xcur, orthant, n);
       
        /* Evaluate the new function and gradient values */
        *fcur = *d.f(n, xcur, cgrad, *d.f_data);
        normx = l1norm_vector(n, xcur); 
        *fcur += normx * (*d.lambda); 
        ++*(stop->nevals_p);    
        ++count;

        /* Check the decrease condition in the paper */
        dgtest = 0.;
        for (i = 0; i < n; ++i){
            dgtest += (xcur[i] - xprev[i]) * pseudograd_prev[i]; 
        }

        if (*fcur <= f_initial + gamma * dgtest){
            return count;
        }

        if (*step < min_step){
            return -2;
        }

        if (*step > max_step){
            return -3;
        }

        if (max_linesearch <= count){
            return -4;
        }

        (*step) *= width;
    }
}




/* The minimization procedure */
nlopt_result owlqn_minimize(int n, nlopt_func f, void *f_data,
                  double *x, /* in: initial guess, out: minimizer */
		          nlopt_stopping *stop, 
                  double *lambda,
                  int m)
{
    owlqn_data d;
    nlopt_result ret = NLOPT_SUCCESS;
    double *work, *xcur, *xprev, *cgrad, *pgrad, 
           *pseudograd, *cgradtmp, fcur, fprev, l1norm, 
           *direction, *gmax, step, *orthant;
    int owlqn_iters = 0;
    int mfv = stop->maxeval;

    int i, ls = 0;
    
    /* Variables for the inverse Hessian update */ 
    int end = 0, k = 1, bound, j;
    double yts, yty, beta;

    
    d.f = f;
    d.f_data = d_data;
    d.lambda = lambda;
    d.stop = stop;

    iteration_data *limited_memory=NULL, *iteration=NULL;
    
    /* Set the number of variables to store if it is 0 */
    if (m<=0){
        m = MAX2(MEMAVAIL/n, 10);
        if (mfv && mfv <= m){
            m = MAX2(mf, 1);
        }
    }

    d.m = m;


    /* must store current point, previous point, current gradient, previous gradient, current pseudogradient, temporary gradient
     * search direction, orthant */
    work = (double *) malloc(sizeof(double) * 8 * n);
    if (!work) return NLOPT_OUT_OF_MEMORY;
    xcur = work;
    xprev = xcur + n;
    cgad = xprev + n;
    pgrad = cgrad + n;
    pseudograd = pgrad + n;
    cgradtmp = pseudograd + n;
    direction = cgradtmp + n;
    orthant = direction + n;
 

    /* Initialize storage */
    memcpy(xcur, x, sizeof(double) * n);
    memcpy(xprev, x, sizeof(double) * n); 

    /* allocate limited memory storage */ 
    limited_memory = (iteration_data*) malloc(sizeof(iteration_data) * m);
    if (!limited_memory) return NLOPT_OUT_OF_MEMORY;

    /* Initialize storage for limited memory */
    for (i = 0; i<m, ++i){
        it = &lm[i];
        it->alpha = 0;
        it->ys = 0;
        it->s = (double*) malloc(sizeof(double) * n);
        it->y = (double*) malloc(sizeof(double) * n);
        if ((!it->s) || (!it->y)) return NLOPT_OUT_OF_MEMORY;
    }
    fprev = HUGE_VAL;
    /********** evaluate function, gradient, and pseudo *************/
    fcur = f(n, xcur, cgrad, f_data);
    
    /* determine the pseudogradient and the first search direction */
    normx = l1norm_vector(n, xcur); 
    fcur += normx * (*d.lambda); 
    ++*(stop->nevals_p);
    pseudo_gradient(pseudograd, xcur, cgrad, n, *d.lambda, gmax);
    memcpy(direction, pseudograd, sizeof(double) * n);
    
    vecdot_owlqn(&step, n, direction, direction);
    step = 1. / sqrt(step);
    
    if (nlopt_stop_time(stop)) {
        ret = NLOPT_MAXTIME_REACHED;
        goto done;
    }
    do {
        /* Check all of the various stopping criteria */ 
        check_stopping_criteria(&n, &owlqn_iters, xcur, xprev,
                &fcur, &fprev, pseudograd, &ret, stop);
        
        if (ret != NLOPT_SUCCESS){
            goto done;
        }
        
        if (owlqn_verbose){
            printf("Objective Value Step %d: %g\n", owlqn_iters, fcur);
        }

        /* Store the current vectors and gradients */
        memcpy(xprev, xcur, sizeof(double) * n);
        memcpy(pgrad, cgrad, sizeof(double) * n);
        fprev = fcur;

        /* Line search for an optimal step */
        ls = line_search_owlqn(n, xcur, &fcur, cgrad, direction, 
                 step, orthant, xprev, pseudograd, &d, stop);
        pseudo_gradient(pseudograd, xcur, cgrad, n, *d.lambda, gmax);  
        
        if (ls < 0){
            memcpy(xcur, xprev, sizeof(double) * n);
            memcpy(cgrad, pgrad, sizeof(double) * n);
            ret = NLOPT_FAILURE;
            if (owlqn_verbose){
                printf("Error in line-search with error code: %d", ls);
            }
            goto done;
        }

        ++owlqn_iters;

        /* Update the vectors s and y that are used for Hessian update */
        iteration = &limited_memory[end];
        vecdiff_owlqn(iteration->s, xcur, xprev, n);
        vecdiff_owlqn(iteration->y, cgrad, pgrad, n);
       
        /* Do the needed dotproducts */
        vecdot_owlqn(&yts, iteration->y, iteration->s, n);
        vecdot_owlqn(&yty, iteration->y, iteration->y, n);
        iteration->ys = yts;


        bound = (m <= k) ? m : k;
        ++k;
        end = (end+1) % m;

        /* steepest direction */ 
        memcpy(direction, pseudograd, sizeof(double) * n);
        
        /* Do the loops for computing new direction from page 799
         * of Nocedal paper */
        j = end;
        for(i = 0; i < bound; ++i){
            j = (j + m - 1) % m;
            iteration = &limited_memory[j];
            vecdot_owlqn(&iteration->alpha, iteration->s, direction, n);
            iteration->alpha /= iteration->ys;
            vecadd_owlqn(direction, iteration->y, -iteration->alpha, n);
        }
        /* We need to scale the direction by H_0 here */
        
        for (i = 0; i < bound; ++i){
            iteration = &limited_memory[j];
            vecdot_owlqn(&beta, iteration->y, direction, n);
            beta /= iteration->ys;
            vecadd_owlqn(direction, iteration->s, iteration->alpha - beta, n);
            j = (j + 1) % m;
        }

        /* Constrain search direction to proper orthant */
        for (i = 0; i < n; ++i){
            if (direction[i] * pseudograd[i] >= 0){
                direction[i] = 0.;
            }
        }
        
        step = 1.0;



    } while (ret == NLOPT_SUCCESS);

done:
    /* remember to free all of the memory */ 
    free(work);
    free(limited_memory);
    return ret;
}









