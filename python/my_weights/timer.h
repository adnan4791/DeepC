/* 
 * $Id: timer.h,v 1.1.1.1 2004-02-06 18:14:56 msato Exp $
 * $RWC_Release$
 * $RWC_Copyright$
 */
/* timer definition */

#ifndef _TIMER_H
#define _TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

double second(void);
double start_timer(void);
double lap_time(char *);

#ifdef __cplusplus
};
#endif

#endif /* _TIMER_H */
