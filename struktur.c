#ifndef STRUKTUR_H
#include "struktur.h"
#endif

int sizeofoutput(taskp parent) {
    return parent->rows * parent->cols * parent->scale;
}

int task_loop(taskp parent) {
    if (parent->a+1 >= parent->b) {
        parent->my_function()
    }
    else {
        int m = (parent->a+parent->b)/2;
        task_t task_left[1];
        task_t task_right[1];
        task_left->a = parent->a;
        task_left->b = m;
        task_left->redobj = parent->redobj;
        task_right->a = m;
        task_right->b = b;
        task_right->redobj = (double *) malloc(rows*cols*scale*sizeof(double));   
        task_loop(task_left);
        task_loop(task_right);
        reduction(task_left,task_right);
    }
}

void reduction(taskp left, taskp right) {
    int xdim = left->cols*left->scale;
    int ydim = left->rows*left->scale;
    for(int i=0;i<ydim;i++)
    {
        for(int j=0;j<xdim;j++)
             left->redobj[i*xdim+j] += right->redobj[i*xdim+j];
    }
}
