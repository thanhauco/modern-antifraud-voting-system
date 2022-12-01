"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Check, ChevronRight, Shield, AlertCircle } from "lucide-react";

interface Candidate {
  id: string;
  name: string;
  party: string;
  image?: string;
}

const candidates: Candidate[] = [
  { id: "cand_1", name: "Jane Smith", party: "Democratic Party" },
  { id: "cand_2", name: "John Doe", party: "Republican Party" },
  { id: "cand_3", name: "Alex Johnson", party: "Independent" },
];

export default function VotePage() {
  const [selectedCandidate, setSelectedCandidate] = useState<string | null>(null);
  const [step, setStep] = useState<"select" | "confirm" | "processing" | "success">("select");

  const handleSubmit = async () => {
    if (!selectedCandidate) return;
    
    setStep("processing");
    
    // Simulate ZK proof generation and blockchain submission
    await new Promise((resolve) => setTimeout(resolve, 3000));
    
    setStep("success");
  };

  return (
    <main className="min-h-screen py-12 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 rounded-full border border-purple-500/30 mb-6">
            <Shield className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-purple-300">Your vote is encrypted and anonymous</span>
          </div>
          <h1 className="text-4xl font-bold mb-4">
            <span className="text-gradient">Washington Governor Election</span>
          </h1>
          <p className="text-white/60">Select your candidate below</p>
        </motion.div>

        {step === "select" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          >
            {candidates.map((candidate, index) => (
              <motion.button
                key={candidate.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setSelectedCandidate(candidate.id)}
                className={`w-full glass-card p-6 flex items-center justify-between group transition-all duration-300
                  ${selectedCandidate === candidate.id 
                    ? "border-purple-500 bg-purple-500/20" 
                    : "hover:bg-white/10"}`}
              >
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-2xl font-bold">
                    {candidate.name.charAt(0)}
                  </div>
                  <div className="text-left">
                    <h3 className="text-xl font-semibold">{candidate.name}</h3>
                    <p className="text-white/60">{candidate.party}</p>
                  </div>
                </div>
                <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center transition-all
                  ${selectedCandidate === candidate.id 
                    ? "border-purple-500 bg-purple-500" 
                    : "border-white/30 group-hover:border-white/50"}`}
                >
                  {selectedCandidate === candidate.id && (
                    <Check className="w-5 h-5 text-white" />
                  )}
                </div>
              </motion.button>
            ))}

            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              onClick={() => selectedCandidate && setStep("confirm")}
              disabled={!selectedCandidate}
              className="w-full btn-primary mt-8 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Continue to Review
              <ChevronRight className="w-5 h-5" />
            </motion.button>
          </motion.div>
        )}

        {step === "confirm" && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card p-8"
          >
            <div className="flex items-start gap-4 mb-6 p-4 bg-yellow-500/10 rounded-xl border border-yellow-500/30">
              <AlertCircle className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-yellow-400 mb-1">Please Review Your Selection</h3>
                <p className="text-white/70 text-sm">
                  Once submitted, your vote cannot be changed. Please verify your selection is correct.
                </p>
              </div>
            </div>

            <div className="py-6 border-b border-white/10">
              <p className="text-white/60 mb-2">Your selection for Governor:</p>
              <p className="text-2xl font-bold">
                {candidates.find((c) => c.id === selectedCandidate)?.name}
              </p>
              <p className="text-white/60">
                {candidates.find((c) => c.id === selectedCandidate)?.party}
              </p>
            </div>

            <div className="flex gap-4 mt-8">
              <button
                onClick={() => setStep("select")}
                className="btn-secondary flex-1"
              >
                Go Back
              </button>
              <button
                onClick={handleSubmit}
                className="btn-primary flex-1"
              >
                Submit Vote
              </button>
            </div>
          </motion.div>
        )}

        {step === "processing" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card p-12 text-center"
          >
            <div className="w-16 h-16 mx-auto mb-6 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
            <h2 className="text-2xl font-bold mb-4">Processing Your Vote</h2>
            <p className="text-white/60 mb-2">Generating zero-knowledge proof...</p>
            <p className="text-white/40 text-sm">This ensures your vote is verifiable yet anonymous</p>
          </motion.div>
        )}

        {step === "success" && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card p-12 text-center"
          >
            <div className="w-20 h-20 mx-auto mb-6 bg-green-500/20 rounded-full flex items-center justify-center">
              <Check className="w-10 h-10 text-green-400" />
            </div>
            <h2 className="text-3xl font-bold mb-4 text-gradient">Vote Submitted Successfully!</h2>
            <p className="text-white/60 mb-8">
              Your vote has been encrypted and recorded on the blockchain.
            </p>
            <div className="glass-card p-4 text-left mb-8">
              <p className="text-white/60 text-sm mb-1">Receipt ID</p>
              <p className="font-mono text-purple-400">receipt_1699123456789</p>
            </div>
            <a href="/verify" className="btn-primary inline-block">
              Verify Your Vote
            </a>
          </motion.div>
        )}
      </div>
    </main>
  );
}
