#include <math.h>
#include <nlopt.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct{
    double *lambda; 
    int samples;
    double * params;
    double *output;
    double ** data;
} function_data;

double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    

    int i, j;
    function_data * my_data = (function_data*) my_func_data;
    int samples = my_data->samples;
    double *params = my_data->params;
    double *output = my_data->output;
    double **data = my_data->data;
    double error = 0.0;
    double error2;
    for (i = 0; i < n; ++i){
        grad[i] = 0.;
    }
    for (i = 0; i < samples; ++i){
        error2 = output[i]; 
        for (j = 0; j < n; ++j){
            error2 -= x[j] * data[i][j];
        }
        error += (error2 * error2);
        for (j = 0; j < n; ++j){
            grad[j] -= 2.0 * data[i][j] * error2;
        }
    }

    return error;
}



double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}



/*
typedef struct {
   double a, b;
} my_constraint_data;

double myconstraint(unsigned n, const double *x, double *grad, void *data)
{
   my_constraint_data *d = (my_constraint_data *) data;
   double a = d->a, b = d->b;
   if (grad){
       grad[0] = 3*a*(a*x[0] + b) * (a*x[0] + b);
       grad[1] = -1.0;
   }
   return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]) ;
}
*/

/* Function to generate noise from linear function 
 * n is the number of features
 * samples is the number of samples in our dataset
 * params is the true value of the parameters
 * data is the data that I have to generate the function 
 * output is where I place the output
 * */ 
void noise_linear_function(int n, int samples, const double *params, double ** data, double * output)
{
    int i, j;

    /* Generate random data */
    for (i = 0; i < samples; ++i){
        for (j = 0; j < n; ++j){
            data[i][j] = randn(0.0, 20.0);      
        }
    }

    for (i = 0; i < samples; ++i){
        output[i] = randn(0.0, 2.0);
        for (j = 0; j < n; ++j){
            output[i] += data[i][j] * params[j];
        }
    }
    return;
}


int main(){
/*    double lb[2] = { -HUGE_VAL, 0};
*/
    int i, j;
    
    double *lambda;
    double lambda_double = 1. ;
    lambda = &lambda_double;
    nlopt_opt opt;
   
    function_data my_data;
    

    int samples = 20;

    int n = 10;


    double ** data = (double **) malloc(samples * sizeof(double*));

    double *output = (double *) malloc(sizeof(double) * samples);
 

    double *params = (double *) malloc(sizeof(double) * n);



    /* Generate random data */
    for (i = 0; i < n; ++i){
        if (i < n/4){
            params[i] = randn(30.0, 1.0);
        }
        else{
            params[i] = randn(0.0, 0.5);
        }
    }

    for (i = 0; i < samples; ++i){
        data[i] = (double *) malloc(n * sizeof(double));
    }


    noise_linear_function(n, samples, params, data, output); 

    my_data.lambda = lambda;
    my_data.samples = samples;
    my_data.output = output;
    my_data.data = data;


     
    void *f_data = (void *) &my_data;
    opt = nlopt_create(NLOPT_LD_OWLQN, n);   
    
    nlopt_set_min_objective(opt, myfunc, f_data);
/*    
    nlopt_set_lower_bounds(opt, lb);
    my_constraint_data data[2] = { {2,0}, {-1,1} };

    nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-8);
    nlopt_add_inequality_constraint(opt, myconstraint, &data[1], 1e-8);
*/
    double *tol = (double *) malloc(sizeof(double));
    *tol = 1e-3;
    nlopt_set_xtol_abs(opt, tol);
    double *x ; 
    x = (double *) malloc(sizeof(double) * n);
    for (i = 0; i < n; ++i){
        x[i] = randn(0.0, 1.0);
    }

    double minf;
    if (nlopt_optimize(opt, x, &minf) < 0) {
        printf("nlopt failed!\n");
        printf("OPT output: %d\n", nlopt_optimize(opt, x, &minf));
    }
    else{
        printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
    }

    nlopt_destroy(opt);
    free(x);
    free(params);
    
    free(data);
    free(output);
    return 0;
}

