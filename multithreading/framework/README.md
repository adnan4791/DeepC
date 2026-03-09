# 🚀 SyncPilot Framework
**Generic Dynamic Load Balancing & Pipeline Reordering Framework for C/C++**

**SyncPilot** is a lightweight C framework specifically designed to execute heavy, multi-staged computational processes (pipelining) utilizing the concepts of an **Asynchronous Worker Pool**, **Priority Queues**, and guaranteed ordered output using a **Reorder Buffer**.

This framework was originally architected to support the performance of the FSRCNN (Fast Super-Resolution Convolutional Neural Network) Deep Learning algorithm on ARM Architectures (big.LITTLE SoC). However, it has now been abstracted to be 100% *generic* and ready to use for other pipelining scenarios, such as:
1. Video / Audio Encoding (FFmpeg-like operations)
2. Graphics Rendering
3. Big Data Processing pipelines
4. Packet Inspection

---

## 🔥 Why Use SyncPilot?
Compared to spawning individual threads for each stage (which triggers bottlenecks when a particular stage is significantly slower), SyncPilot spawns a pool of **Homogeneous Worker Threads**.

1. **Auto-Balancing & Zero Latency Profiling**: If `Stage 3` requires 500ms and `Stage 1` only takes 2ms, the Workers will not sit idle waiting. Automatically (via the *Priority Queue*), workers will immediately swarm `Stage 3` en masse when tasks start to pile up.
2. **Sequential Output without Blocking (Reorder Buffer)**: Because workers are racing against each other, the execution results become out-of-order (*out-of-order execution*). You do not need to worry; SyncPilot reserves a special space called the `Reorder Buffer` that freezes and neatly sorts (*sort*) the results `[0, 1, 2, 3...]` before handing them over to the Final Writer (*Consumer Writer*).
3. **Supports Custom Structs (*void pointers*)**: You are free to throw any struct data shape into this engine because data is passed internally using generic `void *data` pointers.

---

## ⚙️ System Logic Architecture

![SyncPilot Architecture](../docs/hierarki.jpeg)

```text
[ Developer Feed() ] --> [ Stage 0 ] --> [ Stage 1 ] --> [ Stage N (Final) ]
                               \              /                 |
                                \            /                  |
                             [ N Worker Threads ]               v
                                                          [ Reorder Locker ]
                                                                |
 [ Developer Writer() ] <--------------------------------[ Consumer Courier ]
```

---

## 🛠️ Usage Guide (Quick Tutorial)

### 1. Include the *Header* and Define Your Data Structure
```c
#include "framework/syncpilot.h"

// Feel free to create any Struct data type to be passed between Stages!
typedef struct {
    int frame_id;
    char text[256];
    double image_data[1024]; 
} MyPayload;
```

### 2. Define the Stage Processor Functions
Every stage that is called will receive `PipelineTask *task` data. Extract your structure from the `task->data` parameter.

```c
void stage_0_read(PipelineTask *task) {
    MyPayload *payload = (MyPayload*)task->data;
    sprintf(payload->text, "Stage 0 Completed");
    // Perform computation...
}

void stage_1_encode(PipelineTask *task) {
    MyPayload *payload = (MyPayload*)task->data;
    sprintf(payload->text, "Stage 1 Encode Completed");
    // Perform extremely heavy computation...
}
```

### 3. Define the Consumer Writer
This function is special! It is guaranteed by the *framework* to be called **strictly in chronological order**, starting from `Task ID 0`, then `1, 2, 3..`. 
This is where you safely write outputs to a `FILE` / Terminal without data overlapping. Once finished, **you must free the payload memory**.

```c
void final_writer(PipelineTask *task) {
    MyPayload *payload = (MyPayload*)task->data;
    printf("Successfully Saved to Disk! ID: %d | Content: %s\n", task->task_id, payload->text);
    
    free(payload); // FREE YOUR CUSTOM MEMORY!
}
```

### 4. Start the Engine (Main Function)
```c
int main() {
    PipelineConfig cfg;
    cfg.num_workers              = 8;   // Number of concurrent worker threads
    cfg.num_stages               = 2;   // We only have read & encode stages
    cfg.total_tasks              = 100; // Total number of items to be processed
    cfg.queue_capacity_per_stage = 16;  // Stage queue buffer capacity

    // Register your functions
    cfg.stages[0] = stage_0_read;
    cfg.stages[1] = stage_1_encode;
    cfg.consumer  = final_writer;

    // Start the engine!
    PipelineEngine *engine = pipeline_start(&cfg);

    // Throw work into the pipeline!
    for (int i = 0; i < 100; i++) {
        MyPayload *new_payload = (MyPayload*)malloc(sizeof(MyPayload));
        new_payload->frame_id = i;
        
        pipeline_feed(engine, i, new_payload);
    }

    // Signal that the entrance door is closed (No more data)
    pipeline_close_input(engine);

    // Join and wait until completion along with internal API cleanup
    pipeline_wait_and_destroy(engine);
    
    return 0;
}
```

## 📜 Compilation

Because this *framework* natively uses the multitasking synchronization architecture of POSIX Threads, you MUST pass the pthread or OpenMP library *flag*.

```bash
# Example with your main file
gcc -O3 -o my_app main.c framework/syncpilot.c -lpthread -Wall
```

Done! You only need to focus on thinking about the computational math functionality in your C code without ever worrying about complex *deadlocks*, *mutex locks*, *condition signal broadcasts*, or *threadpool* management again!
