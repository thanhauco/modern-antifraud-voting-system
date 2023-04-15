/**
 * Geographic Fraud Detection
 * Detects location-based anomalies like impossible travel, VPN usage, and GPS spoofing
 */

export interface GeoLocation {
  latitude: number;
  longitude: number;
  accuracy: number; // meters
  timestamp: number;
  source: "gps" | "ip" | "wifi";
}

export interface GeoValidationResult {
  isValid: boolean;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  issues: GeoIssue[];
  ipInfo?: IPInfo;
}

export interface GeoIssue {
  type: string;
  description: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
}

export interface IPInfo {
  ip: string;
  country: string;
  region: string;
  city: string;
  isVpn: boolean;
  isProxy: boolean;
  isTor: boolean;
  isDatacenter: boolean;
  riskScore: number;
}

// Washington State boundaries (approximate)
const WA_STATE_BOUNDS = {
  north: 49.0025,
  south: 45.5435,
  east: -116.9165,
  west: -124.8489,
};

export class LocationValidator {
  private previousLocations: Map<string, GeoLocation[]> = new Map();
  private knownVpnRanges: Set<string> = new Set();
  
  constructor() {
    // Load known VPN/datacenter IP ranges (simplified)
    this.loadVpnRanges();
  }

  /**
   * Validates a voter's location for an election
   */
  async validateLocation(
    voterId: string,
    currentLocation: GeoLocation,
    ipAddress: string
  ): Promise<GeoValidationResult> {
    const issues: GeoIssue[] = [];

    // 1. Check if location is within Washington State
    const withinState = this.isWithinWashingtonState(currentLocation);
    if (!withinState) {
      issues.push({
        type: "OUT_OF_JURISDICTION",
        description: "Location is outside Washington State",
        severity: "CRITICAL",
      });
    }

    // 2. Check for impossible travel
    const previousLocs = this.previousLocations.get(voterId) || [];
    const impossibleTravel = this.detectImpossibleTravel(currentLocation, previousLocs);
    if (impossibleTravel) {
      issues.push({
        type: "IMPOSSIBLE_TRAVEL",
        description: impossibleTravel.description,
        severity: "HIGH",
      });
    }

    // 3. Check IP reputation
    const ipInfo = await this.checkIPReputation(ipAddress);
    if (ipInfo.isVpn || ipInfo.isProxy) {
      issues.push({
        type: "VPN_PROXY_DETECTED",
        description: "Connection appears to be through VPN or proxy",
        severity: "MEDIUM",
      });
    }
    if (ipInfo.isTor) {
      issues.push({
        type: "TOR_DETECTED",
        description: "Connection is through Tor network",
        severity: "HIGH",
      });
    }
    if (ipInfo.isDatacenter) {
      issues.push({
        type: "DATACENTER_IP",
        description: "IP belongs to a datacenter/cloud provider",
        severity: "MEDIUM",
      });
    }

    // 4. Check GPS accuracy (very low accuracy may indicate spoofing)
    if (currentLocation.accuracy > 1000) {
      issues.push({
        type: "LOW_GPS_ACCURACY",
        description: `GPS accuracy is ${currentLocation.accuracy}m (poor)`,
        severity: "LOW",
      });
    }

    // 5. Cross-check IP location vs GPS location
    const locationMismatch = await this.checkLocationMismatch(currentLocation, ipAddress);
    if (locationMismatch) {
      issues.push({
        type: "LOCATION_MISMATCH",
        description: "IP geolocation doesn't match GPS location",
        severity: "MEDIUM",
      });
    }

    // Store this location for future checks
    if (!this.previousLocations.has(voterId)) {
      this.previousLocations.set(voterId, []);
    }
    this.previousLocations.get(voterId)!.push(currentLocation);

    // Determine overall risk level
    const riskLevel = this.calculateRiskLevel(issues);

    return {
      isValid: issues.length === 0 || riskLevel === "LOW",
      riskLevel,
      issues,
      ipInfo,
    };
  }

  /**
   * Checks if location is within Washington State
   */
  private isWithinWashingtonState(location: GeoLocation): boolean {
    return (
      location.latitude >= WA_STATE_BOUNDS.south &&
      location.latitude <= WA_STATE_BOUNDS.north &&
      location.longitude >= WA_STATE_BOUNDS.west &&
      location.longitude <= WA_STATE_BOUNDS.east
    );
  }

  /**
   * Detects impossible travel (e.g., voting from Seattle then Los Angeles within 1 hour)
   */
  private detectImpossibleTravel(
    current: GeoLocation,
    previous: GeoLocation[]
  ): { detected: boolean; description: string } | null {
    if (previous.length === 0) return null;

    const lastLocation = previous[previous.length - 1]!;
    const timeDiffHours = (current.timestamp - lastLocation.timestamp) / (1000 * 60 * 60);
    
    // Calculate distance using Haversine formula
    const distance = this.calculateDistance(
      lastLocation.latitude,
      lastLocation.longitude,
      current.latitude,
      current.longitude
    );

    // Maximum reasonable travel speed: ~800 km/h (airplane)
    const maxPossibleDistance = timeDiffHours * 800;

    if (distance > maxPossibleDistance && timeDiffHours < 12) {
      return {
        detected: true,
        description: `Traveled ${Math.round(distance)}km in ${timeDiffHours.toFixed(1)} hours (impossible)`,
      };
    }

    return null;
  }

  /**
   * Calculates distance between two points using Haversine formula
   */
  private calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.toRad(lat2 - lat1);
    const dLon = this.toRad(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.toRad(lat1)) * Math.cos(this.toRad(lat2)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  private toRad(deg: number): number {
    return deg * (Math.PI / 180);
  }

  /**
   * Checks IP reputation using external service (mocked)
   */
  private async checkIPReputation(ipAddress: string): Promise<IPInfo> {
    // In production, this would call an IP reputation service (e.g., IPQualityScore, MaxMind)
    // This is a mock implementation
    const isVpn = this.knownVpnRanges.has(ipAddress.split(".").slice(0, 2).join("."));
    
    return {
      ip: ipAddress,
      country: "US",
      region: "WA",
      city: "Seattle",
      isVpn,
      isProxy: false,
      isTor: ipAddress.startsWith("10."), // Mock Tor detection
      isDatacenter: ipAddress.startsWith("35.") || ipAddress.startsWith("34."), // Mock cloud IPs
      riskScore: isVpn ? 70 : 10,
    };
  }

  /**
   * Checks if IP geolocation matches GPS location
   */
  private async checkLocationMismatch(gpsLocation: GeoLocation, ipAddress: string): Promise<boolean> {
    // In production, compare IP geolocation with GPS
    // This is a simplified mock
    return false;
  }

  /**
   * Calculates overall risk level based on issues
   */
  private calculateRiskLevel(issues: GeoIssue[]): "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" {
    if (issues.some((i) => i.severity === "CRITICAL")) return "CRITICAL";
    if (issues.some((i) => i.severity === "HIGH")) return "HIGH";
    if (issues.some((i) => i.severity === "MEDIUM")) return "MEDIUM";
    return "LOW";
  }

  /**
   * Loads known VPN IP ranges
   */
  private loadVpnRanges(): void {
    // In production, load from a database or external service
    // These are mock ranges
    const mockVpnPrefixes = ["104.238", "45.76", "149.28", "95.179"];
    mockVpnPrefixes.forEach((prefix) => this.knownVpnRanges.add(prefix));
  }
}

export function createLocationValidator(): LocationValidator {
  return new LocationValidator();
}
