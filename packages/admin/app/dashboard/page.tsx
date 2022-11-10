"use client";

import { motion } from "framer-motion";
import { 
  Users, Vote, AlertTriangle, Shield, 
  TrendingUp, Clock, MapPin, Activity 
} from "lucide-react";

// Mock data for dashboard
const stats = [
  { label: "Total Votes", value: "127,845", change: "+12.5%", icon: Vote, color: "purple" },
  { label: "Active Voters", value: "45,231", change: "+8.2%", icon: Users, color: "blue" },
  { label: "Fraud Alerts", value: "23", change: "-15%", icon: AlertTriangle, color: "yellow" },
  { label: "Blocked Attempts", value: "156", change: "+3.1%", icon: Shield, color: "red" },
];

const recentAlerts = [
  { id: 1, type: "GEOGRAPHIC_ANOMALY", severity: "HIGH", time: "2 min ago", details: "Impossible travel detected" },
  { id: 2, type: "BOT_DETECTED", severity: "CRITICAL", time: "5 min ago", details: "Automated voting pattern" },
  { id: 3, type: "DUPLICATE_ATTEMPT", severity: "MEDIUM", time: "12 min ago", details: "Double vote prevented" },
];

export default function DashboardPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-white p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Election Command Center</h1>
        <p className="text-white/60">Washington Governor Election 2024 - Live Monitoring</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-12 h-12 rounded-xl bg-${stat.color}-500/20 flex items-center justify-center`}>
                <stat.icon className={`w-6 h-6 text-${stat.color}-400`} />
              </div>
              <span className={`text-sm ${stat.change.startsWith('+') ? 'text-green-400' : 'text-red-400'}`}>
                {stat.change}
              </span>
            </div>
            <p className="text-3xl font-bold mb-1">{stat.value}</p>
            <p className="text-white/60 text-sm">{stat.label}</p>
          </motion.div>
        ))}
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-8">
        {/* Vote Trend Chart */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="lg:col-span-2 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-purple-400" />
              Vote Trend (24h)
            </h2>
            <div className="flex gap-2">
              <button className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-lg text-sm">1H</button>
              <button className="px-3 py-1 bg-white/5 text-white/60 rounded-lg text-sm">6H</button>
              <button className="px-3 py-1 bg-white/5 text-white/60 rounded-lg text-sm">24H</button>
            </div>
          </div>
          <div className="h-64 flex items-center justify-center border border-dashed border-white/20 rounded-xl">
            <p className="text-white/40">Chart visualization would render here</p>
          </div>
        </motion.div>

        {/* Recent Fraud Alerts */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
        >
          <h2 className="text-xl font-semibold flex items-center gap-2 mb-6">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            Recent Alerts
          </h2>
          <div className="space-y-4">
            {recentAlerts.map((alert) => (
              <div
                key={alert.id}
                className="p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/10 transition-colors cursor-pointer"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-semibold
                    ${alert.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : ''}
                    ${alert.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400' : ''}
                    ${alert.severity === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-400' : ''}
                  `}>
                    {alert.severity}
                  </span>
                  <span className="text-white/40 text-xs flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {alert.time}
                  </span>
                </div>
                <p className="text-sm font-medium">{alert.type.replace(/_/g, ' ')}</p>
                <p className="text-white/60 text-sm">{alert.details}</p>
              </div>
            ))}
          </div>
          <button className="w-full mt-4 py-2 text-center text-purple-400 hover:text-purple-300 text-sm">
            View All Alerts →
          </button>
        </motion.div>
      </div>

      {/* Geographic Distribution */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-8 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
      >
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-6">
          <MapPin className="w-5 h-5 text-cyan-400" />
          Geographic Distribution - Washington State
        </h2>
        <div className="h-80 flex items-center justify-center border border-dashed border-white/20 rounded-xl">
          <p className="text-white/40">Interactive county map would render here</p>
        </div>
      </motion.div>
    </main>
  );
}
