import linecache
import os
import tracemalloc

def start(frameN):
    tracemalloc.start(frameN)

def display_current_memory(topN):
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot, limit=topN)

def display_traceback_memory(topN):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')

    # pick the biggest memory block
    stat = top_stats[0]
    print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
    for line in stat.traceback.format():
            print(line)

def display_top(snapshot, key_type='lineno', limit=20):
    snapshot = snapshot.filter_traces((
    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
    tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
            % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

"""
# ... run your application ...
import subprocess
subprocess.call(['python', 'AnalyzeData.py', '-C', 'MethodsData.xlsx'])

snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
"""
