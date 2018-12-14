import Benchmarker
@profile
def a(s):
    for i in range(0,100000):
        print(s)
    b('alright')
@profile
def b(s):
    for i in range(0,20000):
        print(s)

#Benchmarker.start(5)

print('ok')

#Benchmarker.display_current_memory(10)

#Benchmarker.display_traceback_memory(10)

if __name__ == '__main__':
    a('wow')
