"""
Admin Control Panel API
Role-based access control and administrative functions.
"""

import hashlib
import time
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

class Role(Enum):
    """Admin roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"      # Full access
    ELECTION_ADMIN = "election_admin" # Manage elections
    SECURITY_ADMIN = "security_admin" # Fraud & security
    AUDITOR = "auditor"               # Read-only audit access
    OPERATOR = "operator"             # Operational tasks
    VIEWER = "viewer"                 # Dashboard view only


class Permission(Enum):
    """Granular permissions"""
    # Election management
    ELECTION_CREATE = "election:create"
    ELECTION_MODIFY = "election:modify"
    ELECTION_DELETE = "election:delete"
    ELECTION_VIEW = "election:view"
    
    # Vote management
    VOTE_VIEW = "vote:view"
    VOTE_AUDIT = "vote:audit"
    VOTE_INVALIDATE = "vote:invalidate"
    
    # Fraud management
    FRAUD_VIEW_ALERTS = "fraud:view_alerts"
    FRAUD_ACKNOWLEDGE = "fraud:acknowledge"
    FRAUD_RESOLVE = "fraud:resolve"
    FRAUD_CONFIG = "fraud:config"
    
    # User management
    USER_CREATE = "user:create"
    USER_MODIFY = "user:modify"
    USER_DELETE = "user:delete"
    USER_VIEW = "user:view"
    
    # System
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_METRICS = "system:metrics"


# Role -> Permissions mapping
ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.SUPER_ADMIN: list(Permission),
    Role.ELECTION_ADMIN: [
        Permission.ELECTION_CREATE, Permission.ELECTION_MODIFY, Permission.ELECTION_VIEW,
        Permission.VOTE_VIEW, Permission.VOTE_AUDIT, Permission.FRAUD_VIEW_ALERTS,
        Permission.USER_VIEW, Permission.SYSTEM_METRICS
    ],
    Role.SECURITY_ADMIN: [
        Permission.FRAUD_VIEW_ALERTS, Permission.FRAUD_ACKNOWLEDGE, Permission.FRAUD_RESOLVE,
        Permission.FRAUD_CONFIG, Permission.VOTE_VIEW, Permission.VOTE_AUDIT,
        Permission.VOTE_INVALIDATE, Permission.SYSTEM_LOGS, Permission.SYSTEM_METRICS
    ],
    Role.AUDITOR: [
        Permission.ELECTION_VIEW, Permission.VOTE_VIEW, Permission.VOTE_AUDIT,
        Permission.FRAUD_VIEW_ALERTS, Permission.SYSTEM_LOGS, Permission.SYSTEM_METRICS
    ],
    Role.OPERATOR: [
        Permission.ELECTION_VIEW, Permission.VOTE_VIEW, Permission.FRAUD_VIEW_ALERTS,
        Permission.SYSTEM_METRICS
    ],
    Role.VIEWER: [
        Permission.ELECTION_VIEW, Permission.SYSTEM_METRICS
    ]
}


@dataclass
class AdminUser:
    """An admin user"""
    user_id: str
    email: str
    name: str
    role: Role
    permissions: List[Permission]
    created_at: float
    last_login: Optional[float] = None
    mfa_enabled: bool = True
    active: bool = True
    jurisdiction: Optional[str] = None  # For jurisdiction-scoped admins


@dataclass
class AdminSession:
    """Admin session token"""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ip_address: str
    user_agent: str


@dataclass 
class AuditLogEntry:
    """Audit log for admin actions"""
    log_id: str
    timestamp: float
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str
    success: bool


class AdminAuthService:
    """
    Authentication and authorization for admin users.
    """
    
    def __init__(self):
        self.users: Dict[str, AdminUser] = {}
        self.sessions: Dict[str, AdminSession] = {}
        self.session_timeout = 3600  # 1 hour
        
        # Create default super admin
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create initial super admin"""
        admin = AdminUser(
            user_id="admin_001",
            email="admin@votesystem.gov",
            name="System Administrator",
            role=Role.SUPER_ADMIN,
            permissions=ROLE_PERMISSIONS[Role.SUPER_ADMIN],
            created_at=time.time()
        )
        self.users[admin.user_id] = admin
    
    def authenticate(self, email: str, password_hash: str, 
                     ip_address: str, user_agent: str) -> Optional[AdminSession]:
        """Authenticate admin user"""
        user = self._find_user_by_email(email)
        if not user or not user.active:
            return None
        
        # In production, verify password hash against stored hash
        # Simplified for demo
        
        session = AdminSession(
            session_id=secrets.token_urlsafe(32),
            user_id=user.user_id,
            created_at=time.time(),
            expires_at=time.time() + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session.session_id] = session
        user.last_login = time.time()
        
        return session
    
    def validate_session(self, session_id: str) -> Optional[AdminUser]:
        """Validate session and return user"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        if time.time() > session.expires_at:
            del self.sessions[session_id]
            return None
        
        return self.users.get(session.user_id)
    
    def check_permission(self, user: AdminUser, permission: Permission) -> bool:
        """Check if user has permission"""
        return permission in user.permissions
    
    def _find_user_by_email(self, email: str) -> Optional[AdminUser]:
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def logout(self, session_id: str):
        """Logout and invalidate session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


class AdminControlPanel:
    """
    Admin control panel with role-based access.
    """
    
    def __init__(self):
        self.auth = AdminAuthService()
        self.audit_log: List[AuditLogEntry] = []
    
    def _log_action(self, user_id: str, action: str, resource_type: str,
                    resource_id: str, details: Dict, ip: str, success: bool):
        """Log admin action for audit"""
        entry = AuditLogEntry(
            log_id=f"log_{hashlib.sha256(f'{time.time()}{user_id}'.encode()).hexdigest()[:12]}",
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip,
            success=success
        )
        self.audit_log.append(entry)
    
    def create_admin_user(self, creating_user: AdminUser, 
                          email: str, name: str, role: Role,
                          jurisdiction: str = None) -> Optional[AdminUser]:
        """Create a new admin user"""
        # Check permission
        if not self.auth.check_permission(creating_user, Permission.USER_CREATE):
            return None
        
        # Cannot create user with higher role
        if list(Role).index(role) < list(Role).index(creating_user.role):
            return None
        
        user_id = f"admin_{hashlib.sha256(email.encode()).hexdigest()[:8]}"
        
        new_user = AdminUser(
            user_id=user_id,
            email=email,
            name=name,
            role=role,
            permissions=ROLE_PERMISSIONS[role],
            created_at=time.time(),
            jurisdiction=jurisdiction
        )
        
        self.auth.users[user_id] = new_user
        
        self._log_action(
            creating_user.user_id, "CREATE_USER", "user", user_id,
            {"email": email, "role": role.value}, "", True
        )
        
        return new_user
    
    def modify_thresholds(self, user: AdminUser, thresholds: Dict[str, float]) -> bool:
        """Modify fraud detection thresholds"""
        if not self.auth.check_permission(user, Permission.FRAUD_CONFIG):
            return False
        
        # In production, update actual threshold configuration
        
        self._log_action(
            user.user_id, "MODIFY_THRESHOLDS", "fraud_config", "thresholds",
            thresholds, "", True
        )
        
        return True
    
    def invalidate_vote(self, user: AdminUser, vote_id: str, reason: str) -> bool:
        """Invalidate a vote (requires high privilege)"""
        if not self.auth.check_permission(user, Permission.VOTE_INVALIDATE):
            return False
        
        # In production, mark vote as invalid in database
        
        self._log_action(
            user.user_id, "INVALIDATE_VOTE", "vote", vote_id,
            {"reason": reason}, "", True
        )
        
        return True
    
    def get_audit_log(self, user: AdminUser, 
                      start_time: float = None,
                      end_time: float = None,
                      action_filter: str = None,
                      limit: int = 100) -> List[AuditLogEntry]:
        """Get audit log entries"""
        if not self.auth.check_permission(user, Permission.SYSTEM_LOGS):
            return []
        
        logs = self.audit_log
        
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]
        if action_filter:
            logs = [l for l in logs if action_filter in l.action]
        
        return logs[-limit:]
    
    def get_system_metrics(self, user: AdminUser) -> Optional[Dict]:
        """Get system metrics for dashboard"""
        if not self.auth.check_permission(user, Permission.SYSTEM_METRICS):
            return None
        
        return {
            "total_admins": len(self.auth.users),
            "active_sessions": len(self.auth.sessions),
            "audit_entries": len(self.audit_log),
            "uptime_hours": 24.5,  # Would be calculated
            "system_health": "healthy"
        }


class ElectionManagement:
    """
    Election management functions for admins.
    """
    
    def __init__(self, control_panel: AdminControlPanel):
        self.control_panel = control_panel
        self.elections: Dict[str, Dict] = {}
    
    def create_election(self, user: AdminUser, election_data: Dict) -> Optional[str]:
        """Create a new election"""
        if not self.control_panel.auth.check_permission(user, Permission.ELECTION_CREATE):
            return None
        
        election_id = f"election_{hashlib.sha256(f'{time.time()}'.encode()).hexdigest()[:8]}"
        
        self.elections[election_id] = {
            **election_data,
            "id": election_id,
            "created_by": user.user_id,
            "created_at": time.time(),
            "status": "draft"
        }
        
        self.control_panel._log_action(
            user.user_id, "CREATE_ELECTION", "election", election_id,
            {"name": election_data.get("name")}, "", True
        )
        
        return election_id
    
    def start_election(self, user: AdminUser, election_id: str) -> bool:
        """Start an election (open for voting)"""
        if not self.control_panel.auth.check_permission(user, Permission.ELECTION_MODIFY):
            return False
        
        if election_id not in self.elections:
            return False
        
        self.elections[election_id]["status"] = "active"
        self.elections[election_id]["started_at"] = time.time()
        
        self.control_panel._log_action(
            user.user_id, "START_ELECTION", "election", election_id,
            {}, "", True
        )
        
        return True
    
    def end_election(self, user: AdminUser, election_id: str) -> bool:
        """End an election (close voting)"""
        if not self.control_panel.auth.check_permission(user, Permission.ELECTION_MODIFY):
            return False
        
        if election_id not in self.elections:
            return False
        
        self.elections[election_id]["status"] = "closed"
        self.elections[election_id]["ended_at"] = time.time()
        
        self.control_panel._log_action(
            user.user_id, "END_ELECTION", "election", election_id,
            {}, "", True
        )
        
        return True
    
    def get_election_stats(self, user: AdminUser, election_id: str) -> Optional[Dict]:
        """Get real-time election statistics"""
        if not self.control_panel.auth.check_permission(user, Permission.ELECTION_VIEW):
            return None
        
        if election_id not in self.elections:
            return None
        
        election = self.elections[election_id]
        
        return {
            "election_id": election_id,
            "name": election.get("name"),
            "status": election.get("status"),
            "total_votes": 0,  # Would come from vote service
            "turnout_percentage": 0.0,
            "fraud_alerts": 0,
            "last_vote_at": None
        }


if __name__ == "__main__":
    print("=== Admin Control Panel ===\n")
    
    panel = AdminControlPanel()
    elections = ElectionManagement(panel)
    
    # Get super admin
    admin = panel.auth.users["admin_001"]
    print(f"Logged in as: {admin.name} ({admin.role.value})")
    
    # Create security admin
    security_admin = panel.create_admin_user(
        admin, "security@votesystem.gov", "Security Lead", Role.SECURITY_ADMIN
    )
    print(f"Created: {security_admin.name} ({security_admin.role.value})")
    
    # Create election
    election_id = elections.create_election(admin, {
        "name": "Washington State Governor 2024",
        "jurisdiction": "WA",
        "start_date": "2024-11-05",
        "candidates": ["Candidate A", "Candidate B", "Candidate C"]
    })
    print(f"\nCreated election: {election_id}")
    
    # Start election
    elections.start_election(admin, election_id)
    print("Election started!")
    
    # Get metrics
    metrics = panel.get_system_metrics(admin)
    print(f"\nSystem Metrics: {metrics}")
    
    # Get audit log
    logs = panel.get_audit_log(admin, limit=5)
    print(f"\nRecent Audit Entries: {len(logs)}")
    for log in logs:
        print(f"  - {log.action}: {log.resource_type}/{log.resource_id}")
