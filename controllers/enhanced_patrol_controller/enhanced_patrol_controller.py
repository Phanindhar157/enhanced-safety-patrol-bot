#!/usr/bin/env python3

"""
Simple Safety Patrol Bot Controller for Webots 2025a
Basic line following and obstacle avoidance
"""

from controller import Robot, Motor, Camera, DistanceSensor, LED
import time

def main():
    # Initialize robot
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # Get motors
    left_motor = robot.getDevice("left_wheel_motor")
    right_motor = robot.getDevice("right_wheel_motor")
    
    # Set motor positions to infinity for continuous rotation
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    # Get sensors
    ds_front = robot.getDevice("ds_front")
    ds_left = robot.getDevice("ds_left")
    ds_right = robot.getDevice("ds_right")
    line_left = robot.getDevice("line_sensor_left")
    line_right = robot.getDevice("line_sensor_right")
    
    # Enable sensors
    ds_front.enable(timestep)
    ds_left.enable(timestep)
    ds_right.enable(timestep)
    line_left.enable(timestep)
    line_right.enable(timestep)
    
    # Get LEDs
    led_fire = robot.getDevice("led_fire")
    led_gas = robot.getDevice("led_gas")
    
    print("🤖 Simple Safety Patrol Bot Controller Started")
    print("📊 Basic line following and obstacle avoidance active")
    
    step_count = 0
    
    # Main control loop
    while robot.step(timestep) != -1:
        # Read sensor values
        front_dist = ds_front.getValue()
        left_dist = ds_left.getValue()
        right_dist = ds_right.getValue()
        left_line = line_left.getValue()
        right_line = line_right.getValue()
        
        # Simple obstacle avoidance
        if front_dist < 0.3:
            # Turn away from obstacle
            left_motor.setVelocity(-1.0)
            right_motor.setVelocity(1.0)
        # Line following logic
        elif left_line > 0.1 and right_line > 0.1:
            # On line, go straight
            left_motor.setVelocity(2.0)
            right_motor.setVelocity(2.0)
        elif left_line > 0.1:
            # Line on left, turn left
            left_motor.setVelocity(1.0)
            right_motor.setVelocity(2.0)
        elif right_line > 0.1:
            # Line on right, turn right
            left_motor.setVelocity(2.0)
            right_motor.setVelocity(1.0)
        else:
            # No line detected, search
            left_motor.setVelocity(1.0)
            right_motor.setVelocity(-1.0)
        
        # Simulate fire detection (random)
        if step_count % 200 < 10:
            led_fire.set(1)
        else:
            led_fire.set(0)
        
        # Simulate gas detection (random)
        if step_count % 300 < 5:
            led_gas.set(1)
        else:
            led_gas.set(0)
        
        # Print status every 100 steps
        if step_count % 100 == 0:
            print(f"📊 Patrol Bot Status - Step {step_count}")
            print(f"   🚗 Front Distance: {front_dist:.2f}")
            print(f"   📏 Left Line: {left_line:.2f}, Right Line: {right_line:.2f}")
            print(f"   🔥 Fire LED: {led_fire.get()}")
            print(f"   ⛽ Gas LED: {led_gas.get()}")
            print()
        
        step_count += 1

if __name__ == "__main__":
    main()