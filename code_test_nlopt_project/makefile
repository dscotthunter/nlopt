#-lnlopt -lm
# owlqn.c general.c options.c optimize.c stop.c timer.c mt19937ar.c -o tutorial 
all: simple_test.c 
	gcc -g lasso_owlqn.c -o tutorial -lnlopt -lm 
	./tutorial
clean:
	${RM} tutorial
