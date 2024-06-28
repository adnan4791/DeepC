#ifndef FSRCNN_H
#define FSRCNN_H
#endif
typedef void (*voidfunc)(void *);
typedef struct task {
     voidfunc cfunc;
     double *redobj;
     double *img_in;
     double *weight;
     int robsize;
     int filtersize;
     int a;
     int b;
     int rows;
     int cols;
     int scale;
} task_t;

typedef struct task *taskp;
int task_loop(taskp parent);
