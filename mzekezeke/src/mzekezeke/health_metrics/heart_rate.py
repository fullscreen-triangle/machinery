"""
Heart Rate Analysis Module

Provides comprehensive analysis of heart rate data including:
- Heart Rate Variability (HRV) analysis
- Cardiac rhythm pattern detection
- Stress and recovery indicators
- Exercise intensity zones
- Cardiac health risk assessment
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy import signal, stats
from scipy.signal import find_peaks
import warnings

from .base import HealthMetricAnalyzer


class HeartRateAnalyzer(HealthMetricAnalyzer):
    """
    Comprehensive heart rate and HRV analyzer using established scientific methods.
    """
    
    def __init__(self):
        super().__init__("HeartRateAnalyzer", "1.0.0")
        self.supported_metrics = {"heart_rate", "rr_intervals", "ecg"}
        
        # Clinical reference ranges
        self.resting_hr_ranges = {
            "athletes": (40, 60),
            "excellent": (60, 70),
            "good": (70, 80),
            "average": (80, 90),
            "poor": (90, 100),
            "concerning": (100, float('inf'))
        }
        
        # HRV reference values (RMSSD in ms)
        self.hrv_ranges = {
            "excellent": (50, float('inf')),
            "good": (30, 50),
            "average": (20, 30),
            "poor": (10, 20),
            "very_poor": (0, 10)
        }

    def validate_data(self, data: List[Any]) -> bool:
        """Validate heart rate data."""
        if not data:
            return False
        
        # Check if we have reasonable heart rate values
        for point in data:
            hr_value = point.value if hasattr(point, 'value') else point
            if isinstance(hr_value, (int, float)):
                if not (30 <= hr_value <= 220):  # Physiological HR range
                    return False
            
        return True

    def analyze(self, data: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive heart rate analysis.
        
        Args:
            data: Heart rate data points
            context: Analysis context (age, activity level, etc.)
            
        Returns:
            Comprehensive analysis results
        """
        if not self.validate_data(data):
            return {"error": "Invalid heart rate data"}
        
        # Extract heart rate values and timestamps
        hr_values = self.preprocess_data(data)
        timestamps = [point.timestamp for point in data if hasattr(point, 'timestamp')]
        
        # Basic statistics
        basic_stats = self.calculate_basic_stats(hr_values)
        
        # Heart rate zones analysis
        hr_zones = self.analyze_heart_rate_zones(hr_values, context.get('age', 35))
        
        # Heart rate variability analysis
        hrv_analysis = self.analyze_hrv(hr_values)
        
        # Pattern detection
        patterns = self.detect_cardiac_patterns(hr_values, timestamps)
        
        # Recovery analysis
        recovery = self.analyze_recovery_patterns(hr_values, timestamps)
        
        # Risk assessment
        risk_assessment = self.assess_cardiac_risk(hr_values, hrv_analysis, context)
        
        # Trend analysis
        trend = self.calculate_trend(hr_values, timestamps)
        
        return {
            "metric_type": "heart_rate",
            "analysis_type": "comprehensive_cardiac_analysis",
            "timestamp": datetime.now(),
            "basic_statistics": basic_stats,
            "heart_rate_zones": hr_zones,
            "hrv_analysis": hrv_analysis,
            "cardiac_patterns": patterns,
            "recovery_analysis": recovery,
            "risk_assessment": risk_assessment,
            "trend_analysis": trend,
            "recommendations": self.generate_recommendations(hr_values, hrv_analysis),
            "confidence": self.calculate_analysis_confidence(hr_values),
            "metadata": self.get_metadata()
        }

    def analyze_heart_rate_zones(self, hr_values: np.ndarray, age: int) -> Dict[str, Any]:
        """Analyze time spent in different heart rate zones."""
        max_hr = 220 - age
        
        # Define zones based on % of max HR
        zones = {
            "recovery": (0, 0.6 * max_hr),
            "aerobic_base": (0.6 * max_hr, 0.7 * max_hr),
            "aerobic": (0.7 * max_hr, 0.8 * max_hr),
            "lactate_threshold": (0.8 * max_hr, 0.9 * max_hr),
            "vo2_max": (0.9 * max_hr, max_hr),
            "neuromuscular": (max_hr, float('inf'))
        }
        
        zone_analysis = {}
        total_readings = len(hr_values)
        
        for zone_name, (lower, upper) in zones.items():
            in_zone = np.sum((hr_values >= lower) & (hr_values < upper))
            percentage = (in_zone / total_readings) * 100 if total_readings > 0 else 0
            
            zone_analysis[zone_name] = {
                "count": int(in_zone),
                "percentage": round(percentage, 2),
                "range": (int(lower), int(upper))
            }
        
        return {
            "estimated_max_hr": max_hr,
            "zones": zone_analysis,
            "dominant_zone": max(zone_analysis.keys(), 
                               key=lambda k: zone_analysis[k]["percentage"])
        }

    def analyze_hrv(self, hr_values: np.ndarray) -> Dict[str, Any]:
        """Analyze Heart Rate Variability."""
        if len(hr_values) < 10:
            return {"error": "Insufficient data for HRV analysis"}
        
        # Convert HR to RR intervals (approximation)
        rr_intervals = 60000 / hr_values  # Convert HR to RR intervals in ms
        
        # Time-domain HRV metrics
        time_domain = self.calculate_time_domain_hrv(rr_intervals)
        
        # HRV score
        hrv_score = self.calculate_hrv_score(time_domain)
        
        return {
            "time_domain": time_domain,
            "hrv_score": hrv_score,
            "interpretation": self.interpret_hrv_score(hrv_score)
        }

    def calculate_time_domain_hrv(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate time-domain HRV metrics."""
        rr_diff = np.diff(rr_intervals)
        
        return {
            "mean_rr": float(np.mean(rr_intervals)),
            "sdnn": float(np.std(rr_intervals)),  # Standard deviation of NN intervals
            "rmssd": float(np.sqrt(np.mean(rr_diff ** 2))),  # Root mean square of successive differences
            "pnn50": float(np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100),  # % of successive RR intervals differing by > 50ms
            "cv": float(np.std(rr_intervals) / np.mean(rr_intervals) * 100)  # Coefficient of variation
        }

    def calculate_hrv_score(self, time_domain: Dict[str, float]) -> float:
        """Calculate overall HRV score (0-100)."""
        score = 50  # Start with neutral score
        
        # RMSSD contribution (most important time-domain metric)
        if "rmssd" in time_domain:
            rmssd = time_domain["rmssd"]
            if rmssd >= 50:
                score += 25
            elif rmssd >= 30:
                score += 15
            elif rmssd < 10:
                score -= 20
        
        # SDNN contribution
        if "sdnn" in time_domain:
            sdnn = time_domain["sdnn"]
            if sdnn >= 50:
                score += 15
            elif sdnn >= 30:
                score += 10
            elif sdnn < 20:
                score -= 10
        
        return max(0, min(100, score))

    def interpret_hrv_score(self, score: float) -> str:
        """Interpret HRV score."""
        if score >= 80:
            return "Excellent autonomic function"
        elif score >= 60:
            return "Good autonomic function"
        elif score >= 40:
            return "Average autonomic function"
        elif score >= 20:
            return "Below average autonomic function"
        else:
            return "Poor autonomic function - consider stress management"

    def detect_cardiac_patterns(self, hr_values: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
        """Detect cardiac patterns and irregularities."""
        patterns = {}
        
        # Detect sudden changes
        if len(hr_values) > 5:
            hr_changes = np.abs(np.diff(hr_values))
            sudden_changes = np.where(hr_changes > 20)[0]  # Changes > 20 bpm
            patterns["sudden_changes"] = {
                "count": len(sudden_changes),
                "indices": sudden_changes.tolist(),
                "severity": "high" if len(sudden_changes) > len(hr_values) * 0.1 else "normal"
            }
        
        # Detect sustained high/low periods
        high_hr_threshold = 100
        low_hr_threshold = 50
        
        high_periods = self.find_sustained_periods(hr_values, lambda x: x > high_hr_threshold, min_duration=5)
        low_periods = self.find_sustained_periods(hr_values, lambda x: x < low_hr_threshold, min_duration=5)
        
        patterns["sustained_high_hr"] = high_periods
        patterns["sustained_low_hr"] = low_periods
        
        # Detect irregular rhythm patterns
        if len(hr_values) > 10:
            rr_intervals = 60000 / hr_values
            irregularity_score = self.calculate_rhythm_irregularity(rr_intervals)
            patterns["rhythm_irregularity"] = {
                "score": irregularity_score,
                "interpretation": "irregular" if irregularity_score > 0.3 else "regular"
            }
        
        return patterns

    def find_sustained_periods(self, values: np.ndarray, condition: callable, min_duration: int) -> Dict[str, Any]:
        """Find periods where a condition is sustained."""
        mask = condition(values)
        periods = []
        
        start = None
        for i, is_condition in enumerate(mask):
            if is_condition and start is None:
                start = i
            elif not is_condition and start is not None:
                if i - start >= min_duration:
                    periods.append((start, i - 1))
                start = None
        
        # Handle case where condition extends to end
        if start is not None and len(mask) - start >= min_duration:
            periods.append((start, len(mask) - 1))
        
        return {
            "count": len(periods),
            "periods": periods,
            "total_duration": sum(end - start + 1 for start, end in periods)
        }

    def calculate_rhythm_irregularity(self, rr_intervals: np.ndarray) -> float:
        """Calculate rhythm irregularity score."""
        if len(rr_intervals) < 3:
            return 0.0
        
        # Coefficient of variation of RR intervals
        cv = np.std(rr_intervals) / np.mean(rr_intervals)
        
        # Normalize to 0-1 scale
        return min(cv / 0.5, 1.0)

    def analyze_recovery_patterns(self, hr_values: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze heart rate recovery patterns."""
        recovery = {}
        
        # Find potential recovery periods (decreasing HR trends)
        if len(hr_values) > 10:
            # Use moving average to smooth data
            window_size = min(5, len(hr_values) // 2)
            smoothed_hr = np.convolve(hr_values, np.ones(window_size) / window_size, mode='valid')
            
            # Find periods of sustained decrease
            decreasing_periods = []
            current_start = None
            
            for i in range(1, len(smoothed_hr)):
                if smoothed_hr[i] < smoothed_hr[i-1]:
                    if current_start is None:
                        current_start = i - 1
                else:
                    if current_start is not None and i - current_start >= 3:
                        decreasing_periods.append((current_start, i - 1))
                    current_start = None
            
            recovery["recovery_periods"] = {
                "count": len(decreasing_periods),
                "periods": decreasing_periods
            }
            
            # Calculate average recovery rate
            if decreasing_periods:
                recovery_rates = []
                for start, end in decreasing_periods:
                    rate = (smoothed_hr[start] - smoothed_hr[end]) / (end - start)
                    recovery_rates.append(rate)
                
                recovery["average_recovery_rate"] = float(np.mean(recovery_rates))
                recovery["recovery_quality"] = "good" if np.mean(recovery_rates) > 1.0 else "average"
        
        return recovery

    def assess_cardiac_risk(self, hr_values: np.ndarray, hrv_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cardiac risk based on HR and HRV data."""
        risk_factors = []
        risk_score = 0
        
        # Age-adjusted resting HR assessment
        age = context.get('age', 35)
        mean_hr = np.mean(hr_values)
        
        if mean_hr > 100:
            risk_factors.append("Elevated resting heart rate")
            risk_score += 2
        elif mean_hr > 90:
            risk_score += 1
        
        # HRV assessment
        if "hrv_score" in hrv_analysis:
            hrv_score = hrv_analysis["hrv_score"]
            if hrv_score < 20:
                risk_factors.append("Very low heart rate variability")
                risk_score += 3
            elif hrv_score < 40:
                risk_factors.append("Low heart rate variability")
                risk_score += 1
        
        # Heart rate variability patterns
        if "time_domain" in hrv_analysis:
            rmssd = hrv_analysis["time_domain"].get("rmssd", 0)
            if rmssd < 10:
                risk_factors.append("Very low RMSSD indicating autonomic dysfunction")
                risk_score += 2
        
        # Determine overall risk level
        if risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "moderate"
        elif risk_score >= 1:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendations": self.get_risk_recommendations(risk_level, risk_factors)
        }

    def get_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get recommendations based on risk assessment."""
        recommendations = []
        
        if risk_level == "high":
            recommendations.append("Consult with healthcare provider about cardiac health")
            recommendations.append("Consider comprehensive cardiac evaluation")
        
        if "Elevated resting heart rate" in risk_factors:
            recommendations.append("Focus on cardiovascular fitness improvement")
            recommendations.append("Monitor caffeine and stimulant intake")
        
        if any("heart rate variability" in factor for factor in risk_factors):
            recommendations.append("Practice stress reduction techniques")
            recommendations.append("Ensure adequate sleep quality")
            recommendations.append("Consider meditation or breathing exercises")
        
        return recommendations

    def generate_recommendations(self, hr_values: np.ndarray, hrv_analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        mean_hr = np.mean(hr_values)
        
        # General fitness recommendations
        if mean_hr > 90:
            recommendations.append("Consider increasing cardiovascular exercise")
        elif mean_hr < 50:
            recommendations.append("Monitor for any symptoms with low heart rate")
        
        # HRV-based recommendations
        if "hrv_score" in hrv_analysis:
            hrv_score = hrv_analysis["hrv_score"]
            if hrv_score < 40:
                recommendations.append("Focus on stress management and recovery")
                recommendations.append("Ensure 7-9 hours of quality sleep")
        
        return recommendations

    def calculate_analysis_confidence(self, hr_values: np.ndarray) -> float:
        """Calculate confidence in the analysis results."""
        confidence = 1.0
        
        # Reduce confidence based on data amount
        if len(hr_values) < 10:
            confidence *= 0.5
        elif len(hr_values) < 50:
            confidence *= 0.8
        
        # Reduce confidence if data is too variable (potential noise)
        cv = np.std(hr_values) / np.mean(hr_values)
        if cv > 0.3:  # Very high variability
            confidence *= 0.7
        
        # Reduce confidence for very short time periods
        if len(hr_values) > 1:
            duration = (hr_values[-1] - hr_values[0]).total_seconds()
            if duration < 300:  # Less than 5 minutes
                confidence *= 0.6
        
        return max(0.1, confidence) 