// Anti-fraud analytics package exports
export { BiometricAnalyzer, createBiometricAnalyzer } from "./detection/behavioral/biometric_analyzer";
export { LocationValidator, createLocationValidator } from "./detection/geographic/location_validator";

export type {
  MouseEvent,
  KeystrokeEvent,
  BehavioralProfile,
} from "./detection/behavioral/biometric_analyzer";

export type {
  GeoLocation,
  GeoValidationResult,
  GeoIssue,
  IPInfo,
} from "./detection/geographic/location_validator";
