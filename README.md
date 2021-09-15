# RBD-LAB Drone Project
- Project goal: program a drone to navigate in closed space.
- Video illustration: 

## Algorithm:
```
1. Attempt to find an object waypoint, i.e. an object of prominent color. We chose a red sweatshirt.
   1.1. Once found, approach that object until its close enough.
      1.2. As long as it not found: 
      
         1.2.1. Use the map of features given by ORB_SLAM2. 

         1.2.2. Estimate obstacles nearby, with the observation that:
         Point in Map <----> Feature in frame <---->  Feature of some object <----> Part of an obstacle

         1.2.3. Thus, go to the location nearby, such that it is the most vacant of map points

         1.2.4. Scan enviroment for that object
```

* Attempt to find an object waypoint, i.e. an object of prominent color. We chose a red sweatshirt.

    * Once found, approach that object until its close enough.
        * As long as it not found:
         
            * Use the map of features given by ORB_SLAM2. 
            * Estimate obstacles nearby, with the observation that: 
            - Point in Map <----> Feature in frame <---->  Feature of some object <----> Part of an obstacle
            * Thus, go to the location nearby, such that it is the most vacant of map points
            * Scan enviroment for that object

---
## Group Members
- Reuven Veinblat
- Daniel Rispler
- Lahav Padan
---
## External Projects Used
- [_cpp-python-sockets_](https://github.com/johnathanchiu/cpp-python-sockets)
- [_Color-Tracker_](https://github.com/gaborvecsei/Color-Tracker)
---
## Dependencies
```
pip install opencv-python
pip install numpy
pip install djitellopy
pip install pynput
pip install sklearn
pip install scipy
pip install matplotlib
pip install scipy
pip install pandas
pip install pytest
```

## Requirements
- ORB_SLAM2 & Project are found under ```/home/$USER/```

