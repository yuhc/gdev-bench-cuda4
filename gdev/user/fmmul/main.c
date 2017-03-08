#include <stdio.h>

int cuda_test_fmmul(unsigned int n, char *path);

int main(int argc, char *argv[])
{
	unsigned int n = 1024;

	if (argc > 1)
		n = atoi(argv[1]);

	if (cuda_test_fmmul(n, ".") < 0)
		printf("Test failed\n");
	else
		printf("Test passed\n");
	
	return 0;
}
