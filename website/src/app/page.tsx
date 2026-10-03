"use client";

import { motion } from "framer-motion";
import { ArrowRight, BarChart3, CheckCircle2, ChevronRight, Database, XCircle } from "lucide-react";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 15 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring" as const,
      stiffness: 100,
      damping: 15,
      mass: 1,
    },
  },
};

export default function Home() {
  return (
    <main className="min-h-screen py-16 px-6 sm:px-12 md:px-24 lg:px-48 max-w-7xl mx-auto selection:bg-accent selection:text-foreground">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-24"
      >
        <motion.header variants={itemVariants} className="space-y-6">
          <div className="flex items-center space-x-2 text-muted text-sm font-medium uppercase tracking-widest mb-8">
            <span className="w-8 h-[1px] bg-muted inline-block"></span>
            <span>Research Findings</span>
          </div>
          <h1 className="text-4xl md:text-6xl font-medium tracking-tight leading-tight">
            Trace-to-Specialist
          </h1>
          <p className="text-lg md:text-xl text-muted max-w-2xl leading-relaxed">
            Can we discover the real task distribution from messy AI usage traces, find the cheapest model that handles each task, and specialize a small model until it beats renting frontier intelligence?
          </p>
        </motion.header>

        <motion.section variants={itemVariants} className="space-y-8">
          <div className="flex items-center justify-between border-b border-border-color pb-4">
            <h2 className="text-2xl font-medium tracking-tight">The Approach</h2>
          </div>
          <div className="text-lg text-muted leading-relaxed max-w-3xl space-y-4">
            <p>
              This is a miniature implementation of the enterprise intelligence loop:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Observe:</strong> Mine traces into deterministic tasks without relying on enterprise data.</li>
              <li><strong>Benchmark:</strong> Evaluate models per task on both process (tool usage) and outcome.</li>
              <li><strong>Route:</strong> Find the cheapest model that clears a &ge;90% success threshold.</li>
              <li><strong>Specialize:</strong> Train a small model via SFT and GRPO where volume justifies it.</li>
            </ul>
          </div>
        </motion.section>

        <motion.section variants={itemVariants} className="space-y-8">
          <div className="flex items-center justify-between border-b border-border-color pb-4">
            <h2 className="text-2xl font-medium tracking-tight">The Core Result</h2>
          </div>
          
          <div className="bg-accent/30 rounded-2xl p-8 md:p-12 shadow-refined border border-border-color/50">
            <h3 className="text-xl font-medium mb-4 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-foreground" />
              Financial Metric Extraction
            </h3>
            <p className="text-muted mb-8 leading-relaxed">
              Tested on unseen companies, pulling numbers from SEC filings. The goal was to reach a 90% router threshold to safely offload from frontier models.
            </p>
            
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-border-color text-sm text-muted">
                    <th className="pb-4 font-medium pl-4">Model Scale</th>
                    <th className="pb-4 font-medium">Base Model</th>
                    <th className="pb-4 font-medium">SFT (Supervised)</th>
                    <th className="pb-4 font-medium pr-4">GRPO (RL)</th>
                  </tr>
                </thead>
                <tbody className="text-sm md:text-base">
                  <tr className="border-b border-border-color/50 hover:bg-accent/20 transition-colors">
                    <td className="py-4 pl-4 font-medium">0.8B</td>
                    <td className="py-4 text-muted">4.40 / 0%</td>
                    <td className="py-4 text-muted">2.54 / 0%</td>
                    <td className="py-4 text-muted">2.58 / 0%</td>
                  </tr>
                  <tr className="border-b border-border-color/50 hover:bg-accent/20 transition-colors">
                    <td className="py-4 pl-4 font-medium">1.5B</td>
                    <td className="py-4 text-muted">1.20 / 0%</td>
                    <td className="py-4">9.16 / 88%</td>
                    <td className="py-4">9.16 / 88%</td>
                  </tr>
                  <tr className="bg-accent/40 rounded-lg">
                    <td className="py-4 pl-4 font-medium text-foreground">3B</td>
                    <td className="py-4 text-muted">5.18 / 50%</td>
                    <td className="py-4 font-medium text-foreground">9.76 / 96%</td>
                    <td className="py-4 text-foreground">9.74 / 94%</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="mt-4 text-xs text-muted text-right">
              * Score format: (Avg Reward /10, Success Rate)
            </div>
          </div>
        </motion.section>

        <motion.section variants={itemVariants} className="space-y-8">
          <div className="flex items-center justify-between border-b border-border-color pb-4">
            <h2 className="text-2xl font-medium tracking-tight">Key Insights</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-8 rounded-2xl border border-border-color bg-background shadow-refined group hover:border-foreground/30 transition-colors duration-500">
              <Database className="w-6 h-6 mb-6 text-muted group-hover:text-foreground transition-colors" />
              <h3 className="text-lg font-medium mb-3">The Capacity Floor</h3>
              <p className="text-muted leading-relaxed text-sm md:text-base">
                Exact-copy grounding emerges strictly between 0.8B and 3B parameters. Below this floor, neither SFT nor properly engaged GRPO loops can solve the task. The small model burns money, while the 3B specialist clears the 96% bar.
              </p>
            </div>

            <div className="p-8 rounded-2xl border border-border-color bg-background shadow-refined group hover:border-foreground/30 transition-colors duration-500">
              <BarChart3 className="w-6 h-6 mb-6 text-muted group-hover:text-foreground transition-colors" />
              <h3 className="text-lg font-medium mb-3">SFT Teaches Format, Not Grounding</h3>
              <p className="text-muted leading-relaxed text-sm md:text-base">
                Supervised Fine-Tuning is incredible for format adherence, but it cannot fix grounding. On thin data, SFT can actually hurt performance (a 40-demo run scored 0.0, its twin 5.45) by causing rigid mimicry.
              </p>
            </div>

            <div className="p-8 rounded-2xl border border-border-color bg-background shadow-refined group hover:border-foreground/30 transition-colors duration-500 md:col-span-2">
              <XCircle className="w-6 h-6 mb-6 text-muted group-hover:text-foreground transition-colors" />
              <h3 className="text-lg font-medium mb-3">RL's Missing Magic</h3>
              <p className="text-muted leading-relaxed text-sm md:text-base max-w-3xl">
                Across four distinct regimes (floor, ceiling, compositional middle, fair middle), Reinforcement Learning (GRPO) went 0 for 4. Every residual wall encountered was a capability gap that RL could not sample across at this data scale. At 1.5B, the wall was retrieval misses; RL couldn't fix it. At 3B, it was already at the ceiling.
              </p>
            </div>
          </div>
        </motion.section>

        <motion.section variants={itemVariants} className="pt-8 border-t border-border-color">
          <p className="text-lg font-medium leading-relaxed max-w-3xl">
            Specialization pays above the capacity floor and burns money below it. 
            The 3B model successfully serves as a robust router target, proving that we can offload 
            expensive tasks to cheap, owned specialists.
          </p>
        </motion.section>
        
        <motion.footer variants={itemVariants} className="pt-24 pb-8 flex justify-between items-center text-sm text-muted">
          <span>Trace-to-Specialist &copy; {new Date().getFullYear()}</span>
          <a href="https://ayeangad.xyz" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors duration-300 underline underline-offset-4 decoration-border-color hover:decoration-foreground">
            ayeangad.xyz
          </a>
        </motion.footer>
      </motion.div>
    </main>
  );
}
