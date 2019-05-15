#include <math.h>
#include <nlopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct{
    double *lambda; 
    int samples;
    double *output;
    double ** data;
} function_data;

double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    

    int i, j;
    function_data * my_data = (function_data*) my_func_data;
    int samples = my_data->samples;
    const double *output = my_data->output;
    double **data = my_data->data;
    double neg_log_likelihood = 0.0;
    double value;
    for (i = 0; i < n; ++i){
        grad[i] = 0.;
    }
    for (i = 0; i < samples; ++i){
        value = 0.0; 
        for (j = 0; j < n; ++j){
            value += x[j] * data[i][j];
        }

        if (output[i] > 0.95){ 
            neg_log_likelihood += log( 1.0 + exp(-value) ) ;
        }
        else if(output[i] < 0.05){
            neg_log_likelihood += log( 1.0 + exp(value) ) ;
        }
        else{
            printf("ERROR IN VALUE: %f\n", output[i]);
        }

        for (j = 0; j < n; ++j){
            grad[j] -= ( output[i] - (1.0 / (1.0 + exp( -value )) )) * data[i][j] ;
        }
    }
    
    double error_full = 0.0;
    error_full += neg_log_likelihood;
    for (j = 0; j < n; ++j){
        error_full += *(my_data->lambda) * fabs(x[j]);
    }
    printf("%f\n", error_full);

    return neg_log_likelihood;
}

void get_field(char* line, double *output, double **data, int n, int j)
{
    const char* tok;
    int i = 0;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (i<n){
            data[j][i] = atof(tok);
        }
        else if(i==n){
            output[j] = atof(tok);
        }
        i++;
     
    }
    return;
}

void get_data(int samples, int n, double * output, double **data)
{
    int i;
    int j = 0;
    FILE* stream = fopen("data_as_csv/breast_cancer.csv", "r");
    char line[1024];
    while (fgets(line, 1024, stream))
    { 
        char* tmp = strdup(line);
        get_field(tmp, output, data, n, j);
        free(tmp);
        ++j;
    }
    return;
}



int main(){
    int i, j;
    
    double *lambda;
    double lambda_double = 1. ;
    lambda = &lambda_double;
    nlopt_opt opt;
   
    function_data my_data;
    
    /* For boston data set samples = 506 and n = 13 */
    /* For breast cancer data set samples = 569 and n = 30 */
    int samples = 569;

    int n = 30;


    double ** data = (double **) malloc(samples * sizeof(double*));

    double *output = (double *) malloc(sizeof(double) * samples);
 
    for (i = 0; i < samples; ++i){
        data[i] = (double *) malloc(n * sizeof(double));
    }

    get_data(samples, n, output, data);



    my_data.lambda = lambda;
    my_data.samples = samples;
    my_data.output = output;
    my_data.data = data;


     
    void *f_data = (void *) &my_data;
    opt = nlopt_create(NLOPT_LD_OWLQN, n);   
    
    nlopt_set_min_objective(opt, myfunc, f_data);

    double tol = 59.843537754;
    nlopt_set_stopval(opt, tol);
    double *x ; 
    x = (double *) malloc(sizeof(double) * n);
    for (i = 0; i < n; ++i){
        x[i] = 0.0;
    }

    unsigned M = 2;
    nlopt_set_vector_storage(opt, M);

    double minf;
    int out = nlopt_optimize(opt, x, &minf);
    if (out < 0) {
        printf("nlopt failed!\n");
        printf("OPT output: %d\n", out);
   }
   else{
        printf("OPT output: %d\n", out);
    } 
    printf("%d\n", nlopt_get_vector_storage(opt));
      

    nlopt_destroy(opt);
    free(x);
    
    free(data);
    free(output);
    return 0;
}
