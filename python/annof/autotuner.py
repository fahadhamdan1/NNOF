import tvm
from tvm import te
import annof_core

def example_autotuning():
    @te.hybrid.script
    def matmul_add(a, b, c):
        n = a.shape[0]
        m = c.shape[1]
        k = a.shape[1]
        for i, j in te.grid(n, m):
            sum = c[i, j]
            for kk in te.range(k):
                sum += a[i, kk] * b[kk, j]
            c[i, j] = sum
        return c

    A = te.placeholder((1024, 1024), name="A")
    B = te.placeholder((1024, 1024), name="B")
    C = te.placeholder((1024, 1024), name="C")

    target = tvm.target.Target("opencl")
    task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(A, B, C), target=target)

    log_file = "matmul_add_opencl_log.json"
    tune_option = tvm.auto_scheduler.TuningOptions(
        num_measure_trials=200,
        measure_callbacks=[tvm.auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)

    func = tvm.build(sch, args, target)
    return func