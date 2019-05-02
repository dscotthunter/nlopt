#-lnlopt -lm
# owlqn.c general.c options.c optimize.c stop.c timer.c mt19937ar.c -o tutorial 
all: simple_test.c 
	gcc -g simple_test.c -o tutorial -lnlopt -lm 
	./tutorial
clean:
	${RM} tutorial
