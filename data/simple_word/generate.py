#!/usr/bin/env python3
"""
Generate 5K simple word problems for curriculum SFT.

Categories:
- Addition/Subtraction (1.5K): 1-step add/sub word problems
- Multiplication (1.5K): 1-step multiplication word problems  
- Two-step (2K): Simple 2-step problems combining operations
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer.mul_facts import MulFactTokens

MUL_FACTS = MulFactTokens()


def format_mul_token(a: int, b: int) -> Optional[str]:
    """Return mul-token if applicable, else None."""
    if a > b:
        a, b = b, a
    if 1 <= a <= 9 and 1 <= b <= 9:
        result = a * b
        return f"<MUL_{a}_{b}_{result}>"
    return None


# Templates for word problems
ADDITION_TEMPLATES = [
    ("Alice has {a} apples. Bob gives her {b} more apples. How many apples does Alice have now?",
     "{a} + {b} = {result}. Alice has {result} apples."),
    
    ("There are {a} birds in a tree. {b} more birds fly to the tree. How many birds are in the tree?",
     "{a} + {b} = {result}. There are {result} birds in the tree."),
    
    ("A store has {a} red shirts and {b} blue shirts. How many shirts does the store have in total?",
     "{a} + {b} = {result}. The store has {result} shirts in total."),
    
    ("Tom scored {a} points in the first game and {b} points in the second game. What is his total score?",
     "{a} + {b} = {result}. His total score is {result} points."),
    
    ("A baker made {a} cookies in the morning and {b} cookies in the afternoon. How many cookies did the baker make?",
     "{a} + {b} = {result}. The baker made {result} cookies."),
]

SUBTRACTION_TEMPLATES = [
    ("Sarah has {a} stickers. She gives {b} stickers to her friend. How many stickers does Sarah have left?",
     "{a} - {b} = {result}. Sarah has {result} stickers left."),
    
    ("A jar has {a} candies. {b} candies are eaten. How many candies remain?",
     "{a} - {b} = {result}. {result} candies remain."),
    
    ("There were {a} books on a shelf. {b} books were borrowed. How many books are still on the shelf?",
     "{a} - {b} = {result}. There are {result} books still on the shelf."),
    
    ("A farmer has {a} chickens. {b} chickens are sold. How many chickens does the farmer have now?",
     "{a} - {b} = {result}. The farmer has {result} chickens now."),
    
    ("Mike has {a} dollars. He spends {b} dollars on lunch. How much money does he have left?",
     "{a} - {b} = {result}. He has {result} dollars left."),
]

MULTIPLICATION_TEMPLATES = [
    ("A box contains {a} pencils. There are {b} boxes. How many pencils are there in total?",
     "{a} × {b} = {expr}. There are {result} pencils in total."),
    
    ("Each basket has {a} apples. If there are {b} baskets, how many apples are there altogether?",
     "{a} × {b} = {expr}. There are {result} apples altogether."),
    
    ("A parking lot has {a} rows. Each row has {b} cars. How many cars are in the parking lot?",
     "{a} × {b} = {expr}. There are {result} cars in the parking lot."),
    
    ("A pack of gum has {a} pieces. If you buy {b} packs, how many pieces do you have?",
     "{a} × {b} = {expr}. You have {result} pieces."),
    
    ("Each shelf holds {a} books. The library has {b} shelves. How many books can the library hold?",
     "{a} × {b} = {expr}. The library can hold {result} books."),
    
    ("A teacher gives {a} stickers to each student. There are {b} students. How many stickers does the teacher give out?",
     "{a} × {b} = {expr}. The teacher gives out {result} stickers."),
]

TWO_STEP_TEMPLATES = [
    # Multiplication then subtraction
    ("A store has {a} shelves with {b} toys on each shelf. If {c} toys are sold, how many toys remain?",
     "{a} × {b} = {mul_expr}. {mul_result} - {c} = {final}. {final} toys remain."),
    
    # Multiplication then addition
    ("Emily has {a} boxes of {b} crayons each. Her mom gives her {c} more crayons. How many crayons does Emily have?",
     "{a} × {b} = {mul_expr}. {mul_result} + {c} = {final}. Emily has {final} crayons."),
    
    # Addition then multiplication
    ("John has {a} red marbles and {b} blue marbles. He puts them in {c} equal groups. How many marbles are in each group?",
     "{a} + {b} = {sum_result}. {sum_result} ÷ {c} = {final}. Each group has {final} marbles."),
    
    # Two multiplications
    ("A bakery sells {a} types of cookies. Each type has {b} boxes. Each box has {c} cookies. How many cookies does the bakery have?",
     "{a} × {b} = {mul1_expr}. {mul1_result} × {c} = {final}. The bakery has {final} cookies."),
]


def generate_add_sub(count: int, inject_mul: bool) -> list[dict]:
    """Generate addition and subtraction word problems."""
    examples = []
    
    while len(examples) < count:
        # Split evenly between add and sub
        if len(examples) < count // 2:
            template, answer_template = random.choice(ADDITION_TEMPLATES)
            a = random.randint(5, 50)
            b = random.randint(3, 30)
            result = a + b
        else:
            template, answer_template = random.choice(SUBTRACTION_TEMPLATES)
            a = random.randint(20, 80)
            b = random.randint(5, a - 5)  # Ensure positive result
            result = a - b
        
        question = template.format(a=a, b=b)
        answer = answer_template.format(a=a, b=b, result=result)
        answer = f"{answer}\n#### {result}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "category": "add_sub",
            "steps": 1
        })
    
    return examples


def generate_multiplication(count: int, inject_mul: bool) -> list[dict]:
    """Generate multiplication word problems."""
    examples = []
    
    while len(examples) < count:
        template, answer_template = random.choice(MULTIPLICATION_TEMPLATES)
        
        # Use numbers that can benefit from mul-tokens
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        result = a * b
        
        # Format expression
        if inject_mul:
            mul_token = format_mul_token(a, b)
            expr = mul_token if mul_token else str(result)
        else:
            expr = str(result)
        
        question = template.format(a=a, b=b)
        answer = answer_template.format(a=a, b=b, expr=expr, result=result)
        answer = f"{answer}\n#### {result}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "category": "multiplication",
            "steps": 1
        })
    
    return examples


def generate_two_step(count: int, inject_mul: bool) -> list[dict]:
    """Generate 2-step word problems."""
    examples = []
    
    while len(examples) < count:
        template_idx = random.randint(0, len(TWO_STEP_TEMPLATES) - 1)
        template, answer_template = TWO_STEP_TEMPLATES[template_idx]
        
        if template_idx == 0:  # mul then sub
            a = random.randint(3, 8)
            b = random.randint(3, 8)
            mul_result = a * b
            c = random.randint(2, mul_result - 2)
            final = mul_result - c
            
            if inject_mul:
                mul_token = format_mul_token(a, b)
                mul_expr = mul_token if mul_token else str(mul_result)
            else:
                mul_expr = str(mul_result)
            
            question = template.format(a=a, b=b, c=c)
            answer = answer_template.format(
                a=a, b=b, c=c, mul_expr=mul_expr, mul_result=mul_result, final=final
            )
            
        elif template_idx == 1:  # mul then add
            a = random.randint(3, 7)
            b = random.randint(3, 7)
            mul_result = a * b
            c = random.randint(5, 20)
            final = mul_result + c
            
            if inject_mul:
                mul_token = format_mul_token(a, b)
                mul_expr = mul_token if mul_token else str(mul_result)
            else:
                mul_expr = str(mul_result)
            
            question = template.format(a=a, b=b, c=c)
            answer = answer_template.format(
                a=a, b=b, c=c, mul_expr=mul_expr, mul_result=mul_result, final=final
            )
            
        elif template_idx == 2:  # add then divide
            # Pick c first, then make sum divisible by c
            c = random.randint(2, 6)
            final = random.randint(5, 15)
            sum_result = c * final
            a = random.randint(1, sum_result - 1)
            b = sum_result - a
            
            question = template.format(a=a, b=b, c=c)
            answer = answer_template.format(
                a=a, b=b, c=c, sum_result=sum_result, final=final
            )
            
        else:  # two multiplications
            a = random.randint(2, 5)
            b = random.randint(2, 5)
            c = random.randint(2, 5)
            mul1_result = a * b
            final = mul1_result * c
            
            if inject_mul:
                mul1_token = format_mul_token(a, b)
                mul1_expr = mul1_token if mul1_token else str(mul1_result)
            else:
                mul1_expr = str(mul1_result)
            
            question = template.format(a=a, b=b, c=c)
            answer = answer_template.format(
                a=a, b=b, c=c, mul1_expr=mul1_expr, mul1_result=mul1_result, final=final
            )
        
        answer = f"{answer}\n#### {final}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "category": "two_step",
            "steps": 2
        })
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate simple word problem dataset")
    parser.add_argument("--output-dir", type=str, default="data/simple_word",
                        help="Output directory")
    parser.add_argument("--add-sub-count", type=int, default=1500,
                        help="Addition/subtraction examples (default: 1500)")
    parser.add_argument("--mul-count", type=int, default=1500,
                        help="Multiplication examples (default: 1500)")
    parser.add_argument("--two-step-count", type=int, default=2000,
                        help="Two-step examples (default: 2000)")
    parser.add_argument("--inject-mul-tokens", action="store_true",
                        help="Generate mul_tokens condition version")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    inject = args.inject_mul_tokens
    condition = "mul_tokens" if inject else "baseline"
    
    print(f"Generating simple word problems for condition: {condition}")
    
    all_examples = []
    
    print(f"  Addition/Subtraction: {args.add_sub_count} examples")
    all_examples.extend(generate_add_sub(args.add_sub_count, inject))
    
    print(f"  Multiplication: {args.mul_count} examples")
    all_examples.extend(generate_multiplication(args.mul_count, inject))
    
    print(f"  Two-step: {args.two_step_count} examples")
    all_examples.extend(generate_two_step(args.two_step_count, inject))
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Save
    output_file = out_dir / f"problems_{condition}.json"
    with open(output_file, "w") as f:
        json.dump(all_examples, f, indent=2)
    
    # Save metadata
    metadata = {
        "condition": condition,
        "total_examples": len(all_examples),
        "categories": {
            "add_sub": args.add_sub_count,
            "multiplication": args.mul_count,
            "two_step": args.two_step_count
        },
        "inject_mul_tokens": inject,
        "seed": args.seed
    }
    with open(out_dir / f"metadata_{condition}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved {len(all_examples)} examples to {output_file}")
    
    # Show samples
    print("\n--- Sample examples ---")
    for ex in random.sample(all_examples, 3):
        print(f"Q: {ex['question']}")
        print(f"A: {ex['answer']}")
        print()


if __name__ == "__main__":
    main()

