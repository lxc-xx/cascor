/*       Parity Benchmark Data Set Generator
	 
	 v1.0
	 Matt White  (mwhite+@cmu.edu)
	 9/19/93

	 QUESTIONS/COMMENTS: neural-bench@cs.cmu.edu

	 Usage: a.out <# of inputs>
           # of inputs defaults to 2, or XOR.
	   Care should be taken not to input a number greater than the number
	   of bits, usually 32, of the machine being used, minus one.  Thus,
           for most computers, the maximum input is 31.
  
	 Description
	 ~~~~~~~~~~~
	   This program generates parity benchmark data in the CMU Neural 
         Network Benchmark format. 
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


/*      Function Prototypes      */

void getCommandLine ( int *, int, char ** );
void printHeader    ( int );
void printSet       ( int );


void main  ( int argc, char **argv )
{
  int inputs;    /* Number of inputs into the neural network */   

  getCommandLine ( &inputs, argc, argv );
  printHeader    ( inputs );
  printSet       ( inputs );
}


/*      getCommandLine -  This function reads the values from the command line
	and interprets them.  The only argument should be the number of inputs
        to generate, which defaults to 2.  Extra arguments are ignored.
        Invalid arguments are flagged and an error is produced.
*/

void getCommandLine  ( int *inputs, int argc, char **argv )
{
  if  ( argc < 2 )
    *inputs = 2;
  else  
    if  ( ( **(argv+1) >= '0' ) && ( **(argv+1) <= '9' ) )
      *inputs = atoi( *(argv+1) );
    else  {
      fprintf (stderr, "Invalid argument: %s\n",*(argv+1));
      fprintf (stderr, "Value should be an integer.\n");
      exit( 1 );
    }
}


/*      printHeader -  This function prints out the header information for the
	data set.  This includes generation time (local) and number of inputs.
	It also prints the $SETUP segment for the data set.
*/

void printHeader ( int inputs )
{
  int    i;
  time_t genTime;

  time( &genTime );
  
  printf  (";Parity Benchmark\n");
  printf  (";Generated: %s", asctime( localtime( &genTime ) ));
  printf  (";#Inputs:   %d\n\n", inputs );
  printf  (";Program by: Matt White  (mwhite+@cmu.edu)\n");
  printf  (";  Any questions should be directed to neural-bench@cs.cmu.edu.");
  printf  ("\n\n");

  printf  ("$SETUP\n\n");
  printf  ("PROTOCOL: IO;\n");
  printf  ("OFFSET: 0;\n");
  printf  ("INPUTS: %d;\n", inputs);
  printf  ("OUTPUTS: 1;\n\n");
  for  ( i = 1 ; i <= inputs ; i++ )
    printf  ("IN [%d]: BINARY;\n",i);
  printf  ("OUT [1]: BINARY;\n\n\n");

  printf  ("$TRAIN\n\n");
}


/*      printSet -  Generates and prints out a data set having the specified
	number of inputs.
*/

void printSet  ( int inputs )  
{
  unsigned int num_points,  /* Number of points of data to generate */
               num_pos,     /* Number of positive bits in this number */
               this_point,  /* Number being analyzed */
               i,	    /* General indexing variables */
               j;

  num_points = pow(2,inputs);	/* Figure out how many points to generate */

  for  ( i = 0 ; i < num_points ; i++ )  {
    num_pos = 0;
    this_point = i;
    for  ( j = 0 ; j < inputs ; j++ )  {	/* Analyze the number */
      if  ( j != 0 )
        printf (", ");
      if  ( (this_point & 1) == 1 )  {		/* Get a bit and use it */
        printf ("+");
        num_pos++;
       } else
        printf ("-");
      this_point >>= 1;				/* Shift to the next bit */
    }

    if  ( (num_pos % 2) == 1 )			/* Print the expected output */
      printf  ("  =>  +;\n");
     else
      printf  ("  =>  -;\n"); 
  }  
}





