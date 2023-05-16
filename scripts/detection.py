import pyrealsense2 as rs
import numpy as np
import cv2
import os

##### camera setting
conf = rs.config()
# RGB
conf.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# distance
conf.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# record video
conf.enable_record_to_file('export_filename.bag')  #ADD
# stream start
pipe = rs.pipeline()
profile = pipe.start(conf)

num = 0

while True:
    ############ depth processing ############
    num += 1
    
    ##### getting frame data
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    ##### convert to image
    color_image = np.asanyarray(color_frame.get_data())
    # convert depth to color data
    depth_color_frame = rs.colorizer().colorize(depth_frame)
    depth_image = np.asanyarray(depth_color_frame.get_data())
    cv2.imwrite('./scripts/color_images/color_image-'+str(num)+'.png', color_image)  
    
    ##### filtering to make better depth image data
    # decimarion_filter parameter
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)
    # spatial_filter parameter
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # hole_filling_filter paramter
    hole_filling = rs.hole_filling_filter()
    # disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    # filtering
    filter_frame = decimate.process(depth_frame)
    filter_frame = depth_to_disparity.process(filter_frame)
    filter_frame = spatial.process(filter_frame)
    filter_frame = disparity_to_depth.process(filter_frame)
    filter_frame = hole_filling.process(filter_frame)
    result_frame = filter_frame.as_depth_frame()
    # # convert depth to color data
    # filtered_depth_color_frame = rs.colorizer().colorize(result_frame)
    # filtered_depth_image = np.asanyarray(filtered_depth_color_frame.get_data())
    
    color_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
    
    
    
    ############ Looking for brown box with RGB ############ 
    ##### mask except "brown" part
    # paramter setting for finding "brown"
    BGRlower = np.array([150,150,180])
    BGRhigher = np.array([255,255,255])
    # mask
    masked_image = cv2.inRange(color_image, BGRlower, BGRhigher)
    cv2.imwrite('./scripts/images/mask-image-'+str(num)+'.png', masked_image)  
    # change to gray image
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('/home/slmc/catkin_ws/src/realsense_detection/scripts/images/image-'+str(num)+'.png', gray_image)  
    #ret, bin_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('/home/slmc/catkin_ws/src/realsense_detection/scripts/images/bin-image-'+str(num)+'.png', bin_image)   
    
    
    ##### find rectangle
    contours, hierarchy = cv2.findContours(
        masked_image, 
        cv2.RETR_EXTERNAL,      
        cv2.CHAIN_APPROX_NONE   
        ) 

    largest_area = 0
    
    for i, contour in enumerate(contours):
        # all
        # cv2.drawContours(color_image, contours, i, (255, 0, 0), 2)

        # no-tilted ones only
        # x,y,w,h = cv2.boundingRect(contour)
        # cv2.rectangle(color_image,(x,y),(x+w-1,y+h-1),(0,255,0),2)

        # accept tilted ones
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # record the largest rectangle
        if largest_area < ((box[0][0]-box[1][0])**2 + (box[0][1]-box[2][1])**2):
            largest_area = ((box[0][0]-box[1][0])**2 + (box[0][1]-box[2][1])**2)
            largest_box = box
        
        
        
    ############ Calculate the target box coordination ############ 
    if largest_area != 0:
        # calclate the center of the target in RGB image
        target_box_center_1 = (largest_box[0][0] + largest_box[2][0]) // 2
        target_box_center_2 = (largest_box[0][1] + largest_box[2][1]) // 2
        
        # distance from camera to the center of the box
        target_box_distance = result_frame.get_distance(target_box_center_1, target_box_center_2)
        
        # calculate (x,y,z) of the target box
        target_box_coordination = rs.rs2_deproject_pixel_to_point(color_intr , [target_box_center_1, target_box_center_2], target_box_distance)  
        print("box coordination: ", target_box_coordination)
        
        # draw the detected box
        cv2.drawContours(color_image,[largest_box],0,(0,0,255), 2)
        # print(color_image[360][640])
        cv2.circle(color_image, (target_box_center_1, target_box_center_2), 2, (0,255,0), 5)
        cv2.putText(color_image, str(target_box_distance), (target_box_center_1 + 5, target_box_center_2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Image", color_image)
        cv2.waitKey(200)
        
        


