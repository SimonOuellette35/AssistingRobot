from OrbitMode import OrbitMode
from time import time

orbit = OrbitMode(None)

print("Initializing to leftmost cam servo")
orbit.init_phase(1250)

print("Running detection")
start_time = time()
human_detected, human_coords = orbit.detect_human()
end_time = time()

print("RPC took %s seconds" % (end_time - start_time))
if human_detected:
    print("ERROR ==> Detected a human at coords: %s" % (human_coords))

orbit.close()
exit()