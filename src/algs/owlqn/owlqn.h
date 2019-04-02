#ifndef OWLQN_H
#define OWLQN_H

#include "nlopt.h"
#include "nlopt-util.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */ 

extern int owlqn_verbose;

typedef struct { 
    nlopt_func f; /* This is just the loss function */
    void *f_data;
    int m; /* the number of variables to remember */
    double *lambda, *gradtmp;
    nlopt_stopping *stop;
} owlqn_data;

/* Helper functions */ 
double l1norm_vector(int n, double *x);
void vecdot_owlqn(double *out, double *x, double *y, int n);
void vecadd_owlqn(double *x, double *y, double *c, int n);
void vecdiff_owlqn(double *x, double *y, double *z, int n);
void vecscale_owlqn(double *x, double scale, int n);
void vecnegcopy_owlqn(double *x, double *y, int n);
void owlqn_project(double *x, double *y, int n);

void pseudo_gradient(double* pseudo_grad,
        const double *x,
        const double *grad, 
        const int n,
        const double lambda,
        double *gmax /* L-infinity norm pseudo-gradient */
        );

void check_stopping_criteria(int *n, int *owlqn_iters, 
        double *xcur, double *xprev,
        double *fcur, double *fprev, 
        double *gmax, nlopt_result *ret, 
        nlopt_stopping *stop);

int line_search_owlqn(int n, 
        double *xcur, double *fcur, 
        double *grad, double *direction,
        double *step, double *orthant, 
        const double *xprev, const double *pseudograd_prev, 
        owlqn_data *d, 
        nlopt_stopping *stop);

/* The main function */
nlopt_result owlqn_minimize(int n, nlopt_func f, void *f_data, /* f_data stores lambda */
                  double *x, /* in: initial guess, out: minimizer */
		  nlopt_stopping *stop,
          int m); /* m is the amount of memory */

#define MEMAVAIL 1310720  /* I am using this much because that is what Professor Johnson uses in luksan/ for LBFGS */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* OWLQN_H */
