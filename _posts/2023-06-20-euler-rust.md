---
title: "Project Euler in Rust"
excerpt: "My Project Euler track in Rust."
tags:
    - mathematics
    - algorithmic
    - rust
---

```rust
pub fn problem9() -> i32 {
    for a in 1..998 {
        for b in 1..std::cmp::min(a, 999 - a) {
            let c = 1_000 - a - b;
            if a*a + b*b == c*c { return a*b*c }
        }
    }
    panic!();
}
```

Project Euler provides a set of great exercises to practice algorithmic, but I find the first problems quite good to learn a new language. Hence, I solved the first 30 [problems](https://projecteuler.net) in Rust to practice the language foundations.