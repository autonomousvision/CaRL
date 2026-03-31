import os

local_rank = int(os.environ['LOCAL_RANK'])

# Example: 8 GPUs, 2 NUMA nodes (0–3, 4–7)
if 0 <= local_rank <= 3:
    numa_node = 0
elif 4 <= local_rank <= 7:
    numa_node = 1
else:
    numa_node = None

cmd = ['python'] + os.sys.argv[1:]

if numa_node is not None:
    cmd = [
        'numactl',
        f'--cpunodebind={numa_node}',
        f'--preferred={numa_node}',
    ] + cmd

print('Command:', cmd)
print('Command 0:', cmd[0])

os.execvp(cmd[0], cmd)