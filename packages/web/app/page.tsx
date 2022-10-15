"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Shield, Vote, CheckCircle, Lock, Eye, FileCheck } from "lucide-react";

export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 px-4 overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/30 rounded-full blur-3xl animate-pulse-slow" />
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-pink-500/30 rounded-full blur-3xl animate-pulse-slow" />
        </div>

        <div className="relative max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 rounded-full border border-purple-500/30 mb-8">
              <Shield className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-purple-300">Secured by Blockchain & Post-Quantum Cryptography</span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="text-gradient">Washington State</span>
              <br />
              <span className="text-white">Secure Voting Portal</span>
            </h1>

            <p className="text-xl text-white/70 max-w-2xl mx-auto mb-12">
              Cast your vote with confidence. Our blockchain-based system ensures
              your vote is private, verifiable, and tamper-proof.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/vote" className="btn-primary inline-flex items-center gap-2">
                <Vote className="w-5 h-5" />
                Cast Your Vote
              </Link>
              <Link href="/verify" className="btn-secondary inline-flex items-center gap-2">
                <CheckCircle className="w-5 h-5" />
                Verify Your Vote
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="text-3xl font-bold text-center mb-12 text-gradient"
          >
            Why Trust Our Voting System?
          </motion.h2>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="glass-card p-8 group hover:bg-white/10 transition-all duration-300"
              >
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <feature.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-white/60">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Election Info */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            className="glass-card p-8 md:p-12"
          >
            <h2 className="text-2xl font-bold mb-6 text-gradient">Current Election</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-4 border-b border-white/10">
                <span className="text-white/60">Election</span>
                <span className="font-semibold">Washington Governor 2024</span>
              </div>
              <div className="flex justify-between items-center py-4 border-b border-white/10">
                <span className="text-white/60">Voting Period</span>
                <span className="font-semibold">Nov 5, 2024 - 7:00 AM to 8:00 PM PST</span>
              </div>
              <div className="flex justify-between items-center py-4 border-b border-white/10">
                <span className="text-white/60">Status</span>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-semibold">
                  Active
                </span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </main>
  );
}

const features = [
  {
    icon: Lock,
    title: "Quantum-Resistant Security",
    description: "Protected by NIST-approved post-quantum cryptography algorithms that safeguard against future quantum threats.",
  },
  {
    icon: Eye,
    title: "Anonymous & Verifiable",
    description: "Zero-knowledge proofs ensure your vote is counted correctly while keeping your choice completely private.",
  },
  {
    icon: FileCheck,
    title: "Immutable Record",
    description: "Every vote is recorded on a tamper-proof blockchain, creating an auditable trail without compromising privacy.",
  },
];
