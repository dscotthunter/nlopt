#include <stdlib.h> 
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "owlqn.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

int owlqn_verbose = 1;



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
void vecdot_owlqn(double *out, double *x, double *y, int n){
    int i;
    *out = 0.;
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

void vecnegcopy_owlqn(double *x, double *y, int n){
    int i;
    for (i = 0; i < n; ++i){
        x[i] = -y[i];
    }
    return;
}

/* Scale the vector x by d */
void vecscale_owlqn(double *x, double scale, int n){
    int i;

    for (i = 0; i < n; ++i){
        x[i] *= scale;
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
void pseudo_gradient(double* pseudo_grad,
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
            pseudo_grad[i] = grad[i] - lambda;
        }
        else if (x[i] > 0.){
            pseudo_grad[i] = grad[i] + lambda;
        }
        else{
            /* Not differentiable at this point */
            if (grad[i] < - lambda){
                /* Then right partial is negative so we take that */
                pseudo_grad[i] = grad[i] + lambda;
            }
            else if (lambda < grad[i]){
                /* Then left partial derivative is positive so take that */
                pseudo_grad[i] = grad[i] - lambda;
            }
            else{
                /* otherwise case in paper */
                pseudo_grad[i] = 0.;
            }
        }

        if (i<=0){
            *gmax = fabs(pseudo_grad[i]);
        }
        else if(*gmax >= fabs(pseudo_grad[i])){
            *gmax = fabs(pseudo_grad[i]);
        }
    }
    return;
}





void check_stopping_criteria(int *n, int *owlqn_iters, 
        double *xcur, double *xprev,
        double *fcur, double *fprev, 
        double *gmax, nlopt_result *ret, 
        nlopt_stopping *stop)
{
  if (nlopt_stop_forced(stop)) {
    *ret = NLOPT_FORCED_STOP;
    return;
  }
  if (*fcur <= stop->minf_max){
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
    double gamma = 1e-6;
    double min_step = 1e-40;
    double max_step = 1e20;
    int max_linesearch = 20;

    if (*step <= 0.){
        return -1;
    }

    /* Choose the orthant search direction */
    for (i = 0; i < n; ++i){
        orthant[i] = (xprev[i] == 0.) ? -pseudograd_prev[i] : xprev[i];
    }


    for (;;){
        /* Update the current point */
        memcpy(xcur, xprev, sizeof(double) * n);
        vecadd_owlqn(xcur, direction, step, n);

        /* Project this new point onto the correct orthant */
        owlqn_project(xcur, orthant, n);
       
        /* Evaluate the new function and gradient values */
        *fcur = (*d).f(n, xcur, cgrad, (*d).f_data);
        normx = l1norm_vector(n, xcur); 
        *fcur += normx * (*d->lambda); 
        ++*(stop->nevals_p);    
        ++count;
        /*printf("f_initial - fcur: %g\n", f_initial - *fcur);*/
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
nlopt_result owlqn_minimize(int n, nlopt_func f, void *f_data, /* stores lambda, as well as necessary data for function */
                  double *x, /* in: initial guess, out: minimizer */
		          nlopt_stopping *stop, 
                  int m)
{
    if (owlqn_verbose){
        printf("Initializing Minimize");
    }
    owlqn_data d;
    nlopt_result ret = NLOPT_SUCCESS; 
    double *work, *xcur, *xprev, *cgrad, *pgrad,  *direction, *orthant, *pseudograd; 
    double fcur, fprev, l1norm, step;
    int owlqn_iters = 0;
    int mfv = stop->maxeval;

    int i, ls = 0;
    
    /* Variables for the inverse Hessian update */ 
    int end = 0, k = 1, bound, j;
    double yts, yty, beta;

    double double_tmp;
    double * double_pointer_tmp;

     
    d.f = f;
    d.f_data = f_data;
    d.lambda = ((double**)f_data)[0];
    d.stop = stop;

    iteration_data *limited_memory=NULL, *iteration=NULL;
    
    /* Set the number of variables to store if it is 0 */
    if (m<=0){
        m = MAX(MEMAVAIL/n, 10);
        if (mfv && mfv <= m){
            m = MAX(mfv, 1);
        }
    }

    d.m = m;


    /* must store current point, previous point, current gradient, previous gradient, current pseudogradient, temporary gradient
     * search direction, orthant */
    work = (double *) malloc(sizeof(double) * 7 * n);
    if (!work) return NLOPT_OUT_OF_MEMORY;
    xcur = work;
    xprev = xcur + n;
    cgrad = xprev + n;
    pgrad = cgrad + n;
    pseudograd = pgrad + n; 
    direction = pseudograd + n;
    orthant = direction + n;
 
    double *gmax;
    gmax = (double *) malloc(sizeof(double)); 

    /* Initialize storage */
    memcpy(xcur, x, sizeof(double) * n);
    memcpy(xprev, x, sizeof(double) * n); 

    /* allocate limited memory storage */ 
    limited_memory = (iteration_data*) malloc(sizeof(iteration_data) * m);
    if (!limited_memory) return NLOPT_OUT_OF_MEMORY;

    /* Initialize storage for limited memory */
    for (i = 0; i<m; ++i){
        iteration = &limited_memory[i];
        iteration->alpha = 0;
        iteration->ys = 0;
        iteration->s = (double*) malloc(sizeof(double) * n);
        iteration->y = (double*) malloc(sizeof(double) * n);
        if ((!iteration->s) || (!iteration->y)) return NLOPT_OUT_OF_MEMORY;
    }
    fprev = HUGE_VAL;
    /********** evaluate function, gradient, and pseudo *************/
    fcur = f(n, xcur, cgrad, f_data);
    
    /* determine the pseudogradient and the first search direction */
    l1norm = l1norm_vector(n, xcur); 
    fcur += l1norm * (*d.lambda); 
    ++*(stop->nevals_p); 
    pseudo_gradient(pseudograd, xcur, cgrad, n, *d.lambda, gmax);
    vecnegcopy_owlqn(direction, pseudograd,  n);
    
    vecdot_owlqn(&step, direction, direction, n);
    step = 1. / sqrt(step);
    
    if (nlopt_stop_time(stop)) {
        ret = NLOPT_MAXTIME_REACHED;
        goto done;
    }
    do {
        /* Check all of the various stopping criteria */ 
        check_stopping_criteria(&n, &owlqn_iters, xcur, xprev,
                &fcur, &fprev, gmax, &ret, stop);
        
        if (ret != NLOPT_SUCCESS){
            goto done;
        }
        
        if (owlqn_verbose){
            printf("Objective Value Step %d: %g\n", owlqn_iters, fcur);
        }
        if (*gmax < 1e-4){
            goto done;
        }

        /* Store the current vectors and gradients */
        memcpy(xprev, xcur, sizeof(double) * n);
        memcpy(pgrad, cgrad, sizeof(double) * n);
        fprev = fcur;
        

        /* Line search for an optimal step */
        ls = line_search_owlqn(n, xcur, &fcur, cgrad, direction, 
                 &step, orthant, xprev, pseudograd, &d, stop);
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
        vecnegcopy_owlqn(direction, pseudograd,  n);

        
        
        /* Do the loops for computing new direction from page 799
         * of Nocedal paper */
        j = end;
        for(i = 0; i < bound; ++i){
            j = (j + m - 1) % m;
            iteration = &limited_memory[j];
            vecdot_owlqn(&iteration->alpha, iteration->s, direction, n);
            iteration->alpha /= iteration->ys;
            double_tmp = - iteration->alpha;
            double_pointer_tmp = &double_tmp;
            vecadd_owlqn(direction, iteration->y, double_pointer_tmp, n);
        }
        /* We need to scale the direction by H_0 here */
        vecscale_owlqn(direction, yts / yty, n); 
        
        for (i = 0; i < bound; ++i){
            iteration = &limited_memory[j];
            vecdot_owlqn(&beta, iteration->y, direction, n);
            beta /= iteration->ys;
            double_tmp = iteration->alpha - beta;
            double_pointer_tmp = &double_tmp;
            vecadd_owlqn(direction, iteration->s, double_pointer_tmp, n);
            j = (j + 1) % m;
        }

        /* Constrain search direction to proper orthant */
        for (i = 0; i < n; ++i){
            if (direction[i] * pseudograd[i] >= 0){
                direction[i] = 0.;
            }
        }
        
        memcpy(x, xcur, sizeof(double) * n);
        step = 1.0;



    } while (ret == NLOPT_SUCCESS);

done:
    /* remember to free all of the memory */ 
    free(work);
    free(limited_memory);
    return ret;
}









