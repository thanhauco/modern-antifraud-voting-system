/**
 * Behavioral Biometrics Analyzer
 * Analyzes mouse movement, typing patterns, and session behavior
 * to detect bot activity and distinguish humans from automated systems
 */

export interface MouseEvent {
  x: number;
  y: number;
  timestamp: number;
  type: "move" | "click" | "scroll";
}

export interface KeystrokeEvent {
  key: string;
  timestamp: number;
  duration: number; // key press duration in ms
}

export interface BehavioralProfile {
  mouseEntropy: number;        // Randomness of mouse movements (0-1)
  mouseVelocityVariance: number;
  clickPattern: number;
  keystrokeRhythm: number;     // Typing rhythm consistency
  keystrokeDwellTime: number;  // Average key press duration
  scrollBehavior: number;
  sessionDuration: number;
  isHumanLikely: boolean;
  confidence: number;
}

export class BiometricAnalyzer {
  private mouseEvents: MouseEvent[] = [];
  private keystrokeEvents: KeystrokeEvent[] = [];
  private sessionStart: number;

  constructor() {
    this.sessionStart = Date.now();
  }

  /**
   * Records a mouse event
   */
  recordMouseEvent(event: MouseEvent): void {
    this.mouseEvents.push(event);
    
    // Keep only last 1000 events to prevent memory bloat
    if (this.mouseEvents.length > 1000) {
      this.mouseEvents.shift();
    }
  }

  /**
   * Records a keystroke event
   */
  recordKeystrokeEvent(event: KeystrokeEvent): void {
    this.keystrokeEvents.push(event);
    
    if (this.keystrokeEvents.length > 500) {
      this.keystrokeEvents.shift();
    }
  }

  /**
   * Analyzes collected behavioral data to create a profile
   */
  analyze(): BehavioralProfile {
    const mouseEntropy = this.calculateMouseEntropy();
    const mouseVelocityVariance = this.calculateMouseVelocityVariance();
    const clickPattern = this.calculateClickPattern();
    const keystrokeRhythm = this.calculateKeystrokeRhythm();
    const keystrokeDwellTime = this.calculateKeystrokeDwellTime();
    const scrollBehavior = this.calculateScrollBehavior();
    const sessionDuration = Date.now() - this.sessionStart;

    // Determine if behavior is human-like
    const humanScore = this.calculateHumanScore({
      mouseEntropy,
      mouseVelocityVariance,
      keystrokeRhythm,
      keystrokeDwellTime,
      sessionDuration,
    });

    return {
      mouseEntropy,
      mouseVelocityVariance,
      clickPattern,
      keystrokeRhythm,
      keystrokeDwellTime,
      scrollBehavior,
      sessionDuration,
      isHumanLikely: humanScore > 0.6,
      confidence: humanScore,
    };
  }

  /**
   * Calculates entropy of mouse movement paths
   * Humans have higher entropy (more random) than bots
   */
  private calculateMouseEntropy(): number {
    if (this.mouseEvents.length < 10) return 0;

    const moveEvents = this.mouseEvents.filter((e) => e.type === "move");
    if (moveEvents.length < 10) return 0;

    // Calculate direction changes
    let directionChanges = 0;
    for (let i = 2; i < moveEvents.length; i++) {
      const prev = moveEvents[i - 1]!;
      const prevPrev = moveEvents[i - 2]!;
      const curr = moveEvents[i]!;

      const angle1 = Math.atan2(prev.y - prevPrev.y, prev.x - prevPrev.x);
      const angle2 = Math.atan2(curr.y - prev.y, curr.x - prev.x);

      if (Math.abs(angle1 - angle2) > 0.3) {
        directionChanges++;
      }
    }

    // Normalize to 0-1 range
    return Math.min(1, directionChanges / (moveEvents.length * 0.5));
  }

  /**
   * Calculates variance in mouse velocity
   * Constant velocity suggests automation
   */
  private calculateMouseVelocityVariance(): number {
    if (this.mouseEvents.length < 3) return 0;

    const velocities: number[] = [];
    for (let i = 1; i < this.mouseEvents.length; i++) {
      const prev = this.mouseEvents[i - 1]!;
      const curr = this.mouseEvents[i]!;
      
      const dx = curr.x - prev.x;
      const dy = curr.y - prev.y;
      const dt = curr.timestamp - prev.timestamp || 1;
      
      const velocity = Math.sqrt(dx * dx + dy * dy) / dt;
      velocities.push(velocity);
    }

    if (velocities.length < 2) return 0;

    const mean = velocities.reduce((a, b) => a + b, 0) / velocities.length;
    const variance = velocities.reduce((acc, v) => acc + (v - mean) ** 2, 0) / velocities.length;
    
    // Normalize
    return Math.min(1, variance / 10);
  }

  /**
   * Analyzes click patterns for regularity
   */
  private calculateClickPattern(): number {
    const clicks = this.mouseEvents.filter((e) => e.type === "click");
    if (clicks.length < 2) return 0.5;

    const intervals: number[] = [];
    for (let i = 1; i < clicks.length; i++) {
      intervals.push(clicks[i]!.timestamp - clicks[i - 1]!.timestamp);
    }

    if (intervals.length < 2) return 0.5;

    // Check variance - too consistent suggests bot
    const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / intervals.length;
    const cv = Math.sqrt(variance) / mean; // Coefficient of variation

    // Humans typically have CV > 0.3 for click intervals
    return Math.min(1, cv);
  }

  /**
   * Analyzes typing rhythm
   */
  private calculateKeystrokeRhythm(): number {
    if (this.keystrokeEvents.length < 5) return 0.5;

    const intervals: number[] = [];
    for (let i = 1; i < this.keystrokeEvents.length; i++) {
      intervals.push(this.keystrokeEvents[i]!.timestamp - this.keystrokeEvents[i - 1]!.timestamp);
    }

    const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / intervals.length;
    const cv = Math.sqrt(variance) / mean;

    // Humans have variable typing rhythm (CV typically 0.2-0.5)
    return Math.min(1, cv * 2);
  }

  /**
   * Calculates average key press duration
   */
  private calculateKeystrokeDwellTime(): number {
    if (this.keystrokeEvents.length === 0) return 0;
    
    const totalDuration = this.keystrokeEvents.reduce((acc, e) => acc + e.duration, 0);
    return totalDuration / this.keystrokeEvents.length;
  }

  /**
   * Analyzes scroll behavior
   */
  private calculateScrollBehavior(): number {
    const scrolls = this.mouseEvents.filter((e) => e.type === "scroll");
    if (scrolls.length < 2) return 0;

    // Calculate scroll velocity variance
    const intervals: number[] = [];
    for (let i = 1; i < scrolls.length; i++) {
      intervals.push(scrolls[i]!.timestamp - scrolls[i - 1]!.timestamp);
    }

    if (intervals.length === 0) return 0;

    const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / intervals.length;
    
    return Math.min(1, variance / 10000);
  }

  /**
   * Calculates overall human likelihood score
   */
  private calculateHumanScore(metrics: {
    mouseEntropy: number;
    mouseVelocityVariance: number;
    keystrokeRhythm: number;
    keystrokeDwellTime: number;
    sessionDuration: number;
  }): number {
    let score = 0;
    let weights = 0;

    // Mouse entropy (weight: 3)
    if (metrics.mouseEntropy > 0.3) {
      score += 3;
    } else if (metrics.mouseEntropy > 0.1) {
      score += 1.5;
    }
    weights += 3;

    // Velocity variance (weight: 2)
    if (metrics.mouseVelocityVariance > 0.1) {
      score += 2;
    } else if (metrics.mouseVelocityVariance > 0.05) {
      score += 1;
    }
    weights += 2;

    // Keystroke rhythm (weight: 2)
    if (metrics.keystrokeRhythm > 0.2 && metrics.keystrokeRhythm < 0.8) {
      score += 2;
    } else if (metrics.keystrokeRhythm > 0.1) {
      score += 1;
    }
    weights += 2;

    // Dwell time (weight: 1) - humans typically 80-200ms
    if (metrics.keystrokeDwellTime > 60 && metrics.keystrokeDwellTime < 250) {
      score += 1;
    }
    weights += 1;

    // Session duration (weight: 2) - too fast is suspicious
    if (metrics.sessionDuration > 10000) {
      score += 2;
    } else if (metrics.sessionDuration > 5000) {
      score += 1;
    }
    weights += 2;

    return score / weights;
  }

  /**
   * Resets all collected data
   */
  reset(): void {
    this.mouseEvents = [];
    this.keystrokeEvents = [];
    this.sessionStart = Date.now();
  }
}

// Factory function
export function createBiometricAnalyzer(): BiometricAnalyzer {
  return new BiometricAnalyzer();
}
