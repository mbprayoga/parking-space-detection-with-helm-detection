import threading
import time

def task():
    print(f"Thread {threading.current_thread().name} is starting.")
    time.sleep(5)
    print(f"Thread {threading.current_thread().name} is ending.")

# Create and start threads
thread1 = threading.Thread(target=task, name="Worker-1")
thread2 = threading.Thread(target=task, name="Worker-2")

thread1.start()
thread2.start()

# List active threads
print("Currently running threads:")
for thread in threading.enumerate():
    print(f" - {thread.name}")

# Wait for threads to complete
thread1.join()
thread2.join()

print("Currently running threads:")
for thread in threading.enumerate():
    print(f" - {thread.name}")
