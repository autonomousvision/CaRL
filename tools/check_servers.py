import subprocess

ID = 'ubuntu'

starts = ['\33[41m', '\33[42m']
end = '\33[0m'

for i, server in enumerate([215, 208, 216]):
    try:
        r1, r2 = subprocess.check_output(['ssh', ID + '@134.2.168.' + str(server),
                                          "top -b -n 1 && echo 326556532989993853566541285721 && nvidia-smi -q -d PIDS,UTILIZATION,MEMORY"]).decode(
            'utf-8').split('326556532989993853566541285721')
    except:
        continue
    res = r2.split('\n') + ['GPU']
    D = {}
    for line in r1.split('\n'):
        line = line.split()
        if (len(line) > 2):
            D[line[0]] = line[1]
    print(f'Server: a100-{i}')
    print('=' * 50)
    names = []
    gpu = -1
    mem = -1
    j = 0

    for idx, i in enumerate(res):
        if i == '    FB Memory Usage':
            total = int(res[idx + 1].split()[-2])
            if ('Used' in res[idx + 2]):
                used = int(res[idx + 2].split()[-2])
            else:
                used = int(res[idx + 3].split()[-2])

            mem = (used * 100) // total

        if i == '    Utilization':
            gpu = int(res[idx + 10].split()[-2])
        if 'Process ID' in i:
            try:
                names.append(D[i.split()[-1]])
            except:
                names.append('unknown')
        if i[:3] == 'GPU' and mem != -1:
            start = starts[gpu + mem == 0]
            print(f'GPU {j}: ' + '|' + start + '#' * (gpu // 5) + ' ' * (
                        20 - gpu // 5) + end + '|' + '   ' + 'MEMORY: ' + '|' + start + '#' * (mem // 5) + ' ' * (
                              20 - mem // 5) + end + '| ', ' '.join(names))
            j += 1
            names = []
    print()
