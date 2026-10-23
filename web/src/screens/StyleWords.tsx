import { useEffect, useState } from "react";
import { ApiError, getStyleWords, putStyle } from "../api";
import "../components/Form.css";
import "./Sections.css";
import "./StyleWords.css";
import type { ScreenProps } from "./props";

const MAX_WORDS = 5;

export default function StyleWords({ userId, advance }: ScreenProps) {
  const [rows, setRows] = useState<string[][] | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getStyleWords()
      .then((r) => live && setRows(r.rows))
      .catch((err) => {
        if (live)
          setError(
            err instanceof ApiError ? err.message : "Could not load the words.",
          );
      });
    return () => {
      live = false;
    };
  }, []);

  function toggle(word: string) {
    setPicked((p) =>
      p.includes(word)
        ? p.filter((w) => w !== word)
        : p.length < MAX_WORDS
          ? [...p, word]
          : p,
    );
  }

  async function submit() {
    if (busy || picked.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      await putStyle(userId, picked);
      advance();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not save that. Try again.",
      );
      setBusy(false);
    }
  }

  const full = picked.length >= MAX_WORDS;
  return (
    <div className="form-screen style-screen">
      <h2>Which words sound like your style?</h2>
      <p className="lede">
        Tap up to five. This is the Stated You: it only knows the words you pick
        here.
      </p>
      <fieldset className="chips">
        <legend>
          {picked.length} of up to {MAX_WORDS}
        </legend>
        <div className="style-rows">
          {(rows ?? []).map((row, i) => (
            <div className="chips-row" key={i}>
              {row.map((w) => {
                const on = picked.includes(w);
                return (
                  <button
                    key={w}
                    type="button"
                    className="chip"
                    aria-pressed={on}
                    disabled={full && !on}
                    onClick={() => toggle(w)}
                  >
                    {w}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </fieldset>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="form-actions">
        <button
          className="btn"
          type="button"
          onClick={submit}
          disabled={busy || picked.length === 0}
        >
          {busy ? "Saving…" : "Continue"}
        </button>
      </div>
    </div>
  );
}
