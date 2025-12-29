#!/usr/bin/env python3
"""
Prompt templates for TinyGSM code-to-natural-language conversion.

Version 3: Optimized for maximum mul-token extraction.
"""

SYSTEM_PROMPT = """You are a math teacher creating step-by-step solutions.

CRITICAL: For EVERY multiplication, you MUST show the calculation as: a × b = result

RULES:
1. Single-digit multiplication (1-9 × 1-9): Write EXACTLY as "a × b = result"
   - CORRECT: "3 × 7 = 21" 
   - WRONG: "3 × 7 is 21" or "3×7" or "(3 × 7)"
   
2. Multi-digit numbers: Break into single-digit pieces
   - 45 × 8: Break as "4 × 8 = 32" and "5 × 8 = 40" then combine
   - 365 × 6: Break as "3 × 6 = 18" and "6 × 6 = 36" and "5 × 6 = 30"

3. ALWAYS use this exact format: "a × b = result" (with spaces and equals sign)

4. Keep steps SHORT. One calculation per step when possible.

5. final_answer must be ONLY a number.

OUTPUT: Valid JSON with "steps" array and "final_answer" string."""


USER_PROMPT_TEMPLATE = """Convert this Python code to step-by-step math.

IMPORTANT: Write every single-digit multiplication as "a × b = result" format.

Question: {question}

Code:
{code}

Rules:
- Format: "3 × 7 = 21" (not "3×7=21" or "(3 × 7)")
- Break multi-digit: 24 × 5 → show "4 × 5 = 20" and "2 × 5 = 10" 
- final_answer = just the number"""


FEW_SHOT_EXAMPLES = """
Example 1:
Question: Tom has 7 boxes with 9 pencils each. How many pencils?
Code: boxes=7; pencils=9; total=boxes*pencils; return total

{
  "steps": [
    "Tom has 7 boxes with 9 pencils each.",
    "Total = 7 × 9 = 63."
  ],
  "final_answer": "63"
}

Example 2:  
Question: A store sells 45 shirts at $8 each. Total revenue?
Code: shirts=45; price=8; revenue=shirts*price; return revenue

{
  "steps": [
    "Revenue = 45 × 8.",
    "Break down: 40 × 8 and 5 × 8.",
    "4 × 8 = 32, so 40 × 8 = 320.",
    "5 × 8 = 40.",
    "Total = 320 + 40 = 360."
  ],
  "final_answer": "360"
}

Example 3:
Question: A farm has 6 rows with 8 trees each, plus 3 more trees. Total?
Code: rows=6; per_row=8; extra=3; total=rows*per_row+extra; return total

{
  "steps": [
    "Trees in rows = 6 × 8 = 48.",
    "Extra trees = 3.",
    "Total = 48 + 3 = 51."
  ],
  "final_answer": "51"
}

Now convert:
"""


# Example of ideal output with mul-tokens
IDEAL_EXAMPLE = """
Input: 24 rows × 8 trees per row

Ideal output steps:
1. "Total = 24 × 8."
2. "Break 24 = 20 + 4."
3. "20 × 8: First 2 × 8 = 16, then 16 × 10 = 160."
4. "4 × 8 = 32."
5. "Total = 160 + 32 = 192."

This gives us: <MUL_2_8_16> and <MUL_4_8_32>
"""
