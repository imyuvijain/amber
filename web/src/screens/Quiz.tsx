import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getIpip, putIpip, type IpipItem } from "../api";
import "../components/Form.css";
import "./Quiz.css";
import type { ScreenProps } from "./props";

// The five Mini-IPIP response options, scored 1..5.
const SCALE = [
  "Very inaccurate",
  "Moderately inaccurate",
  "Neither",
  "Moderately accurate",
  "Very accurate",
] as const;

export default function Quiz({ userId, advance }: ScreenProps) {
  const [items, setItems] = useState<IpipItem[] | null>(null);
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const answersRef = useRef(answers);
  answersRef.current = answers;

  useEffect(() => {
    let live = true;
    getIpip()
      .then((r) => live && setItems(r.items))
      .catch((err) => {
        if (live)
          setError(
            err instanceof ApiError ? err.message : "Could not load the quiz.",
          );
      });
    return () => {
      live = false;
    };
  }, []);

  const submit = useCallback(
    async (final: Record<string, number>) => {
      setBusy(true);
      setError(null);
      try {
        await putIpip(userId, final);
        advance();
      } catch (err) {
        setError(
          err instanceof ApiError
            ? err.message
            : "Could not save your answers. Try again.",
        );
        setBusy(false);
      }
    },
    [userId, advance],
  );

  const pick = useCallback(
    (value: number) => {
      if (!items || busy) return;
      const item = items[index];
      const next = { ...answersRef.current, [item.id]: value };
      setAnswers(next);
      if (index < items.length - 1) setIndex(index + 1);
      else void submit(next);
    },
    [items, index, busy, submit],
  );

  // Keys 1-5 answer, like tapping.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const n = Number(e.key);
      if (Number.isInteger(n) && n >= 1 && n <= SCALE.length) pick(n);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pick]);

  const total = items?.length ?? 0;
  const item = items?.[index];
  const done = item !== undefined && index === total - 1 && answers[item.id];
  return (
    <div className="form-screen quiz-screen">
      <h2>How well does this describe you?</h2>
      <p className="lede">
        Describe yourself as you generally are now, not as you wish to be. This
        is the Big Five You: it only knows these answers.
      </p>
      {item && (
        <section className="quiz-card" aria-live="polite">
          <p className="quiz-progress">
            {index + 1} / {total}
          </p>
          <p className="quiz-statement">{item.text}</p>
          <div className="quiz-scale" role="group" aria-label="Your answer">
            {SCALE.map((label, i) => (
              <button
                key={label}
                type="button"
                className="quiz-option"
                aria-pressed={answers[item.id] === i + 1}
                disabled={busy}
                onClick={() => pick(i + 1)}
              >
                <span className="quiz-key" aria-hidden="true">
                  {i + 1}
                </span>
                {label}
              </button>
            ))}
          </div>
        </section>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="form-actions">
        <button
          className="btn btn-quiet"
          type="button"
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          disabled={busy || index === 0}
        >
          Back
        </button>
        {error && done ? (
          <button className="btn" type="button" onClick={() => submit(answers)}>
            Try again
          </button>
        ) : null}
        <button
          className="link"
          type="button"
          onClick={() => submit(answers)}
          disabled={busy || !items}
        >
          {busy ? "Saving…" : "Skip quiz"}
        </button>
      </div>
    </div>
  );
}
