#!/usr/bin/env python3

"""
Emergency Response Controller for Enhanced Safety Patrol Bot
Handles emergency situations, evacuation routes, and crisis management
"""

import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class EmergencyType(Enum):
    FIRE = "fire"
    GAS_LEAK = "gas_leak"
    STRUCTURAL_COLLAPSE = "structural_collapse"
    MEDICAL_EMERGENCY = "medical_emergency"
    SECURITY_BREACH = "security_breach"
    EQUIPMENT_FAILURE = "equipment_failure"

class EvacuationStatus(Enum):
    NORMAL = "normal"
    PREPARING = "preparing"
    EVACUATING = "evacuating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class EmergencyEvent:
    timestamp: str
    emergency_type: EmergencyType
    severity: int  # 1-10 scale
    location: Tuple[float, float, float]
    description: str
    affected_areas: List[str]
    estimated_evacuation_time: int  # seconds
    required_resources: List[str]

@dataclass
class EvacuationRoute:
    route_id: str
    start_location: Tuple[float, float, float]
    end_location: Tuple[float, float, float]
    waypoints: List[Tuple[float, float, float]]
    safety_level: int  # 1-5 scale
    capacity: int
    estimated_time: int  # seconds
    obstacles: List[Tuple[float, float, float]]

class EmergencyResponseController:
    """Advanced emergency response and evacuation management"""
    
    def __init__(self):
        self.active_emergencies = []
        self.evacuation_routes = []
        self.evacuation_status = EvacuationStatus.NORMAL
        self.emergency_protocols = {}
        self.resource_availability = {}
        self.communication_queue = queue.Queue()
        
        # Initialize emergency protocols
        self._init_emergency_protocols()
        self._init_evacuation_routes()
        self._init_resource_management()
        
        print("🚨 Emergency Response Controller initialized")
    
    def _init_emergency_protocols(self):
        """Initialize emergency response protocols"""
        self.emergency_protocols = {
            EmergencyType.FIRE: {
                "immediate_actions": [
                    "Activate fire suppression system",
                    "Alert all personnel",
                    "Begin evacuation",
                    "Contact fire department"
                ],
                "evacuation_priority": 1,
                "safety_radius": 50.0,  # meters
                "required_ppe": ["fire_retardant_suit", "breathing_apparatus"],
                "estimated_response_time": 300  # seconds
            },
            EmergencyType.GAS_LEAK: {
                "immediate_actions": [
                    "Shut off gas supply",
                    "Ventilate area",
                    "Evacuate immediately",
                    "Contact hazmat team"
                ],
                "evacuation_priority": 1,
                "safety_radius": 100.0,
                "required_ppe": ["gas_mask", "protective_suit"],
                "estimated_response_time": 180
            },
            EmergencyType.STRUCTURAL_COLLAPSE: {
                "immediate_actions": [
                    "Secure area",
                    "Evacuate surrounding areas",
                    "Assess structural integrity",
                    "Contact structural engineers"
                ],
                "evacuation_priority": 2,
                "safety_radius": 75.0,
                "required_ppe": ["hard_hat", "safety_vest"],
                "estimated_response_time": 600
            },
            EmergencyType.MEDICAL_EMERGENCY: {
                "immediate_actions": [
                    "Provide first aid",
                    "Call medical services",
                    "Clear evacuation route",
                    "Monitor vital signs"
                ],
                "evacuation_priority": 3,
                "safety_radius": 10.0,
                "required_ppe": ["medical_gloves", "face_mask"],
                "estimated_response_time": 120
            }
        }
    
    def _init_evacuation_routes(self):
        """Initialize predefined evacuation routes"""
        self.evacuation_routes = [
            EvacuationRoute(
                route_id="main_exit_1",
                start_location=(0, 0, 0),
                end_location=(10, 0, 0),
                waypoints=[(2, 0, 0), (5, 0, 0), (8, 0, 0)],
                safety_level=5,
                capacity=100,
                estimated_time=120,
                obstacles=[]
            ),
            EvacuationRoute(
                route_id="emergency_exit_1",
                start_location=(0, 0, 0),
                end_location=(0, 0, 10),
                waypoints=[(0, 0, 3), (0, 0, 6), (0, 0, 8)],
                safety_level=4,
                capacity=50,
                estimated_time=90,
                obstacles=[]
            ),
            EvacuationRoute(
                route_id="secondary_exit_1",
                start_location=(0, 0, 0),
                end_location=(-10, 0, 0),
                waypoints=[(-2, 0, 0), (-5, 0, 0), (-8, 0, 0)],
                safety_level=3,
                capacity=75,
                estimated_time=150,
                obstacles=[]
            )
        ]
    
    def _init_resource_management(self):
        """Initialize resource availability tracking"""
        self.resource_availability = {
            "fire_suppression_systems": 5,
            "gas_masks": 20,
            "breathing_apparatus": 10,
            "first_aid_kits": 15,
            "emergency_vehicles": 2,
            "medical_personnel": 3,
            "fire_department_contact": True,
            "medical_services_contact": True
        }
    
    def detect_emergency(self, sensor_data: Dict, emergency_type: EmergencyType, 
                        severity: int, location: Tuple[float, float, float]) -> EmergencyEvent:
        """Detect and create emergency event"""
        
        emergency = EmergencyEvent(
            timestamp=datetime.now().isoformat(),
            emergency_type=emergency_type,
            severity=severity,
            location=location,
            description=f"{emergency_type.value} detected at severity level {severity}",
            affected_areas=self._calculate_affected_areas(location, emergency_type),
            estimated_evacuation_time=self._estimate_evacuation_time(emergency_type, severity),
            required_resources=self.emergency_protocols[emergency_type]["required_ppe"]
        )
        
        self.active_emergencies.append(emergency)
        
        # Log emergency
        print(f"🚨 EMERGENCY DETECTED: {emergency.description}")
        print(f"   Location: {location}")
        print(f"   Severity: {severity}/10")
        print(f"   Affected Areas: {emergency.affected_areas}")
        
        return emergency
    
    def _calculate_affected_areas(self, location: Tuple[float, float, float], 
                                emergency_type: EmergencyType) -> List[str]:
        """Calculate areas affected by emergency"""
        safety_radius = self.emergency_protocols[emergency_type]["safety_radius"]
        
        # Simple area calculation based on radius
        affected_areas = []
        if location[0] < 0:
            affected_areas.append("west_wing")
        else:
            affected_areas.append("east_wing")
        
        if location[2] < 0:
            affected_areas.append("south_section")
        else:
            affected_areas.append("north_section")
        
        return affected_areas
    
    def _estimate_evacuation_time(self, emergency_type: EmergencyType, severity: int) -> int:
        """Estimate evacuation time based on emergency type and severity"""
        base_time = self.emergency_protocols[emergency_type]["estimated_response_time"]
        severity_multiplier = 1.0 + (severity - 5) * 0.2  # Adjust based on severity
        return int(base_time * severity_multiplier)
    
    def initiate_evacuation(self, emergency: EmergencyEvent) -> bool:
        """Initiate evacuation procedure"""
        try:
            print(f"🚨 INITIATING EVACUATION for {emergency.emergency_type.value}")
            
            # Update evacuation status
            self.evacuation_status = EvacuationStatus.PREPARING
            
            # Select best evacuation route
            best_route = self._select_evacuation_route(emergency)
            
            if not best_route:
                print("❌ No safe evacuation route available!")
                self.evacuation_status = EvacuationStatus.FAILED
                return False
            
            # Execute evacuation protocol
            success = self._execute_evacuation_protocol(emergency, best_route)
            
            if success:
                self.evacuation_status = EvacuationStatus.COMPLETED
                print("✅ Evacuation completed successfully")
            else:
                self.evacuation_status = EvacuationStatus.FAILED
                print("❌ Evacuation failed")
            
            return success
            
        except Exception as e:
            print(f"💥 Error during evacuation: {e}")
            self.evacuation_status = EvacuationStatus.FAILED
            return False
    
    def _select_evacuation_route(self, emergency: EmergencyEvent) -> Optional[EvacuationRoute]:
        """Select the best evacuation route for the emergency"""
        available_routes = []
        
        for route in self.evacuation_routes:
            # Check if route is safe and accessible
            if self._is_route_safe(route, emergency):
                # Calculate route score based on safety, capacity, and time
                score = self._calculate_route_score(route, emergency)
                available_routes.append((route, score))
        
        if not available_routes:
            return None
        
        # Select route with highest score
        best_route = max(available_routes, key=lambda x: x[1])[0]
        print(f"📍 Selected evacuation route: {best_route.route_id}")
        return best_route
    
    def _is_route_safe(self, route: EvacuationRoute, emergency: EmergencyEvent) -> bool:
        """Check if evacuation route is safe given the emergency"""
        # Check if route is affected by emergency
        emergency_radius = self.emergency_protocols[emergency.emergency_type]["safety_radius"]
        
        for waypoint in route.waypoints:
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(waypoint, emergency.location)))
            if distance < emergency_radius:
                return False
        
        # Check for obstacles
        if route.obstacles:
            return False
        
        return True
    
    def _calculate_route_score(self, route: EvacuationRoute, emergency: EmergencyEvent) -> float:
        """Calculate route score for selection"""
        # Factors: safety level, capacity, time, distance from emergency
        safety_score = route.safety_level * 20
        capacity_score = min(route.capacity / 10, 10)
        time_score = max(0, 10 - route.estimated_time / 30)
        
        # Distance from emergency (farther is better)
        start_distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(route.start_location, emergency.location)))
        distance_score = min(start_distance / 10, 10)
        
        total_score = safety_score + capacity_score + time_score + distance_score
        return total_score
    
    def _execute_evacuation_protocol(self, emergency: EmergencyEvent, route: EvacuationRoute) -> bool:
        """Execute the evacuation protocol"""
        try:
            print(f"🚶 Executing evacuation via {route.route_id}")
            
            # Step 1: Alert all personnel
            self._alert_personnel(emergency)
            
            # Step 2: Clear evacuation route
            self._clear_evacuation_route(route)
            
            # Step 3: Guide evacuation
            self._guide_evacuation(route)
            
            # Step 4: Verify evacuation completion
            success = self._verify_evacuation_completion(emergency)
            
            return success
            
        except Exception as e:
            print(f"💥 Error executing evacuation protocol: {e}")
            return False
    
    def _alert_personnel(self, emergency: EmergencyEvent):
        """Alert all personnel about emergency"""
        print(f"📢 ALERTING PERSONNEL: {emergency.description}")
        print(f"   Evacuation required in: {emergency.affected_areas}")
        print(f"   Estimated time: {emergency.estimated_evacuation_time} seconds")
        
        # In a real system, this would send alerts via:
        # - PA system
        # - Mobile notifications
        # - Email alerts
        # - Visual alarms
    
    def _clear_evacuation_route(self, route: EvacuationRoute):
        """Clear obstacles from evacuation route"""
        print(f"🧹 Clearing evacuation route: {route.route_id}")
        
        # Check for obstacles and clear them
        for obstacle in route.obstacles:
            print(f"   Removing obstacle at {obstacle}")
            # In a real system, this would coordinate with:
            # - Security systems
            # - Maintenance teams
            # - Automated systems
    
    def _guide_evacuation(self, route: EvacuationRoute):
        """Guide personnel along evacuation route"""
        print(f"🧭 Guiding evacuation along {route.route_id}")
        
        for i, waypoint in enumerate(route.waypoints):
            print(f"   Waypoint {i+1}: {waypoint}")
            # In a real system, this would:
            # - Activate directional lighting
            # - Play audio guidance
            # - Update digital signage
            # - Provide mobile navigation
    
    def _verify_evacuation_completion(self, emergency: EmergencyEvent) -> bool:
        """Verify that evacuation is complete"""
        print("✅ Verifying evacuation completion...")
        
        # Check if all personnel have evacuated affected areas
        evacuated_areas = emergency.affected_areas
        print(f"   Evacuated areas: {evacuated_areas}")
        
        # In a real system, this would check:
        # - Personnel tracking systems
        # - Access control logs
        # - Visual confirmation
        # - Emergency response team reports
        
        return True  # Simplified for simulation
    
    def manage_emergency_resources(self, emergency: EmergencyEvent) -> Dict[str, bool]:
        """Manage and allocate emergency resources"""
        required_resources = emergency.required_resources
        resource_status = {}
        
        print(f"🔧 Managing resources for {emergency.emergency_type.value}")
        
        for resource in required_resources:
            if resource in self.resource_availability:
                if self.resource_availability[resource] > 0:
                    self.resource_availability[resource] -= 1
                    resource_status[resource] = True
                    print(f"   ✅ {resource}: Available")
                else:
                    resource_status[resource] = False
                    print(f"   ❌ {resource}: Not available")
            else:
                resource_status[resource] = False
                print(f"   ❌ {resource}: Not in inventory")
        
        return resource_status
    
    def coordinate_emergency_response(self, emergency: EmergencyEvent) -> Dict[str, Any]:
        """Coordinate comprehensive emergency response"""
        response_plan = {
            "emergency_id": emergency.timestamp,
            "emergency_type": emergency.emergency_type.value,
            "severity": emergency.severity,
            "location": emergency.location,
            "response_actions": [],
            "resource_allocation": {},
            "evacuation_plan": {},
            "communication_plan": {},
            "timeline": []
        }
        
        # Immediate response actions
        immediate_actions = self.emergency_protocols[emergency.emergency_type]["immediate_actions"]
        response_plan["response_actions"] = immediate_actions
        
        # Resource allocation
        response_plan["resource_allocation"] = self.manage_emergency_resources(emergency)
        
        # Evacuation plan
        best_route = self._select_evacuation_route(emergency)
        if best_route:
            response_plan["evacuation_plan"] = {
                "route_id": best_route.route_id,
                "estimated_time": best_route.estimated_time,
                "capacity": best_route.capacity,
                "safety_level": best_route.safety_level
            }
        
        # Communication plan
        response_plan["communication_plan"] = {
            "internal_alerts": ["all_personnel", "management", "security"],
            "external_contacts": ["emergency_services", "regulatory_authorities"],
            "status_updates": "every_5_minutes"
        }
        
        # Timeline
        response_plan["timeline"] = [
            {"time": 0, "action": "Emergency detected"},
            {"time": 30, "action": "Alert personnel"},
            {"time": 60, "action": "Begin evacuation"},
            {"time": emergency.estimated_evacuation_time, "action": "Evacuation complete"}
        ]
        
        return response_plan
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency status"""
        return {
            "active_emergencies": len(self.active_emergencies),
            "evacuation_status": self.evacuation_status.value,
            "available_resources": self.resource_availability,
            "emergency_types": [e.emergency_type.value for e in self.active_emergencies],
            "last_update": datetime.now().isoformat()
        }
    
    def clear_emergency(self, emergency_id: str) -> bool:
        """Clear resolved emergency"""
        try:
            # Find and remove emergency
            for i, emergency in enumerate(self.active_emergencies):
                if emergency.timestamp == emergency_id:
                    self.active_emergencies.pop(i)
                    print(f"✅ Emergency {emergency_id} cleared")
                    return True
            
            print(f"❌ Emergency {emergency_id} not found")
            return False
            
        except Exception as e:
            print(f"💥 Error clearing emergency: {e}")
            return False

def main():
    """Test the emergency response controller"""
    print("🚨 Testing Emergency Response Controller...")
    
    # Create controller
    controller = EmergencyResponseController()
    
    # Simulate emergency
    emergency = controller.detect_emergency(
        sensor_data={},
        emergency_type=EmergencyType.FIRE,
        severity=8,
        location=(2.0, 0.0, 1.0)
    )
    
    # Coordinate response
    response_plan = controller.coordinate_emergency_response(emergency)
    print(f"\n📋 Response Plan: {json.dumps(response_plan, indent=2)}")
    
    # Initiate evacuation
    success = controller.initiate_evacuation(emergency)
    print(f"\n🚶 Evacuation success: {success}")
    
    # Get status
    status = controller.get_emergency_status()
    print(f"\n📊 Emergency Status: {json.dumps(status, indent=2)}")

if __name__ == "__main__":
    main()

