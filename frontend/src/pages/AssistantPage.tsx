import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  AlertCircle,
  ArrowRight,
  Bot,
  Building,
  CheckCircle2,
  Compass,
  Layers,
  Loader2,
  Send,
  Sparkles,
  Trash2,
  User,
  Wrench,
} from 'lucide-react';
import { sendChatMessage } from '../services/api';
import { ChatMessage } from '../types';
import { Header } from '../components/Header';
import { DisclaimerBanner } from '../components/DisclaimerBanner';

interface PromptCardData {
  category: string;
  title: string;
  prompt: string;
  icon: React.ReactNode;
}

const PROMPT_CARDS: PromptCardData[] = [
  {
    category: 'COMMUNITY ANALYSIS',
    title: 'Tallur Classification Rationale',
    prompt: 'Explain why Tallur has its current priority classification.',
    icon: <Compass className="w-4 h-4 text-bronze" />,
  },
  {
    category: 'DISTRICT COMPARISON',
    title: 'Dakshina Kannada vs Udupi',
    prompt: 'Compare Dakshina Kannada and Udupi.',
    icon: <Building className="w-4 h-4 text-bronze" />,
  },
  {
    category: 'POLICY PRIORITIZATION',
    title: 'Band 1 Critical Allocation',
    prompt: 'Find Band 1 communities in the study area.',
    icon: <AlertCircle className="w-4 h-4 text-[#E05D52]" />,
  },
  {
    category: 'METHODOLOGY AUDIT',
    title: 'Uttara Kannada Fallback',
    prompt: "Explain Uttara Kannada's fallback methodology.",
    icon: <Layers className="w-4 h-4 text-bronze" />,
  },
];

export const AssistantPage: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (queryText?: string) => {
    const textToSend = queryText || inputValue.trim();
    if (!textToSend || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Build conversation history from existing turns
      const historyToSend = messages;
      const response = await sendChatMessage({
        query: textToSend,
        history: historyToSend,
      });

      const toolRecords = response.tool_records || response.tool_executions || [];
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        tool_executions: toolRecords,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `**Request Error:** ${
          err instanceof Error ? err.message : 'Failed to reach assistant orchestrator.'
        }\n\n*Deterministic fallback remains available on the server.*`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const isConversationEmpty = messages.length === 0;

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-12">
      {/* Top Header */}
      <Header
        title="Preparedness Intelligence Assistant"
        subtitle="Conversational interface grounded in 7 deterministic tools, closed-domain scientific guardrails, and authoritative AHP metrics."
        badge="Tool-Grounded"
        actions={
          messages.length > 0 ? (
            <button
              onClick={clearChat}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-taupe hover:text-ivory border border-chocolate rounded hover:bg-chocolate transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Reset Session
            </button>
          ) : undefined
        }
      />

      <DisclaimerBanner
        title="Assessment Scope & Tool Grounding Notice"
        text="This conversational assistant executes in closed-domain mode with strict deterministic tool grounding. Model-generated explanations are bounded by authoritative AHP composite scores and operational priority bands. The backend uses the existing model integration with automated deterministic fallback."
        variant="info"
      />

      {/* Main AI Workspace Canvas */}
      <div className="min-h-[500px] flex flex-col justify-between">
        {isConversationEmpty ? (
          /* Empty State: Editorial Welcome & Prompt Cards Grid */
          <div className="space-y-8 py-6">
            <div className="text-center max-w-2xl mx-auto space-y-3">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-chocolate/60 border border-mocha text-bronze font-mono text-[11px] uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5" />
                Evidence-Grounded Spatial Intelligence
              </div>
              <h2 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-ivory">
                Inquire into Regional Preparedness
              </h2>
              <p className="text-xs sm:text-[13px] text-taupe leading-relaxed">
                Pose spatial inquiries across 1,879 surveyed revenue units in coastal Karnataka. Responses synthesize deterministic AHP indices, Census 2011 infrastructure amenities, and 7 registered analysis tools.
              </p>
            </div>

            {/* Prompt Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              {PROMPT_CARDS.map((card, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleSendMessage(card.prompt)}
                  className="group cursor-pointer p-4 rounded-lg bg-espresso border border-chocolate hover:border-bronze/60 hover:bg-chocolate/40 transition-all flex flex-col justify-between text-left"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-taupe-muted">
                        {card.category}
                      </span>
                      {card.icon}
                    </div>
                    <h3 className="font-display text-sm font-semibold text-ivory group-hover:text-bronze-light transition-colors">
                      {card.title}
                    </h3>
                    <p className="text-xs text-taupe leading-relaxed">
                      {card.prompt}
                    </p>
                  </div>
                  <div className="pt-3 mt-3 border-t border-chocolate/50 flex items-center justify-between text-[11px] text-bronze font-medium w-full">
                    <span>Execute prompt</span>
                    <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Active Conversation Stream */
          <div className="space-y-6 pb-6">
            {messages.map((msg, idx) => {
              const isUser = msg.role === 'user';
              return (
                <div
                  key={idx}
                  className={`flex gap-3.5 max-w-4xl ${isUser ? 'ml-auto flex-row-reverse' : ''}`}
                >
                  {/* Avatar */}
                  <div
                    className={`w-8 h-8 rounded flex items-center justify-center flex-shrink-0 text-xs font-semibold ${
                      isUser
                        ? 'bg-mocha border border-walnut text-ivory'
                        : 'bg-chocolate border border-mocha text-bronze'
                    }`}
                  >
                    {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>

                  {/* Message Bubble */}
                  <div className="space-y-1.5 max-w-[88%]">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-taupe">
                        {isUser ? 'You' : 'Preparedness Intelligence Assistant'}
                      </span>
                      {msg.timestamp && (
                        <span className="text-[10px] text-taupe-muted font-mono tabular-nums">
                          {msg.timestamp}
                        </span>
                      )}
                    </div>

                    <div
                      className={`p-5 rounded-lg text-xs sm:text-[13px] leading-relaxed ${
                        isUser
                          ? 'bg-chocolate border border-mocha text-ivory'
                          : 'bg-espresso border border-chocolate text-ivory shadow-sm'
                      }`}
                    >
                      <div className="prose prose-invert prose-xs max-w-none prose-p:my-2 prose-headings:my-2.5 prose-headings:text-ivory prose-table:my-3 prose-th:p-2 prose-td:p-2 prose-th:bg-chocolate prose-td:border-chocolate font-sans">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>

                      {/* Tool Telemetry Audit Plaque */}
                      {msg.tool_executions && msg.tool_executions.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-chocolate space-y-2">
                          <div className="text-[11px] font-mono text-taupe flex items-center gap-1.5">
                            <Wrench className="w-3.5 h-3.5 text-bronze" />
                            Grounded Deterministic Tools Executed ({msg.tool_executions.length}):
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {msg.tool_executions.map((tool, tIdx) => (
                              <span
                                key={tIdx}
                                className="text-[10px] font-mono px-2 py-0.5 rounded bg-chocolate border border-mocha text-bronze flex items-center gap-1"
                              >
                                <CheckCircle2 className="w-3 h-3 text-[#48A976]" />
                                {tool.tool_name}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}

            {isLoading && (
              <div className="flex gap-3.5 max-w-xl">
                <div className="w-8 h-8 rounded bg-chocolate border border-mocha flex items-center justify-center text-bronze flex-shrink-0">
                  <Bot className="w-4 h-4 animate-pulse" />
                </div>
                <div className="p-4 rounded-lg bg-espresso border border-chocolate text-xs text-taupe flex items-center gap-2.5">
                  <Loader2 className="w-4 h-4 text-bronze animate-spin" />
                  <span>Executing deterministic tools and synthesizing guidance...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Integrated Floating Message Composer */}
        <div className="sticky bottom-4 z-20 pt-4">
          <div className="bg-espresso/95 backdrop-blur-md border border-chocolate rounded-lg p-2.5 shadow-2xl">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="flex items-center gap-2"
            >
              <input
                type="text"
                placeholder="Ask an operational inquiry (e.g., 'Explain why Tallur is Band 4' or 'Compare DK and Udupi')..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                disabled={isLoading}
                className="flex-1 text-xs sm:text-[13px] bg-roast border border-chocolate rounded px-4 py-2.5 text-ivory placeholder-taupe-muted focus:outline-none focus:border-bronze disabled:opacity-50 font-sans"
              />
              <button
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                className="px-4 py-2.5 rounded bg-bronze hover:bg-bronze-light text-roast font-semibold text-xs transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 flex-shrink-0"
                aria-label="Send message"
              >
                <span>Send</span>
                <Send className="w-3.5 h-3.5" />
              </button>
            </form>
            <div className="flex justify-between items-center text-[10px] text-taupe-muted mt-2 px-1 font-mono">
              <span>Deterministic AHP assessments hold absolute operational precedence.</span>
              <span>7 Deterministic Tools Registered</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
