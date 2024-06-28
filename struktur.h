typedef void (*voidfunc)(void *);
typedef int (*intfunc)();

typedef struct thread {
      double *tls;
      int thread_id;
}thread_t;

typedef struct task {
       thread_t *thr;
       double *redobj;
       voidfunc my_function;
       int a;
       int b;
       double *weight;
       double  *filtersize;
       int rows;
       int cols;
       int scale;

} task_t;

typedef struct task *taskp;

voidfunc my_function;


