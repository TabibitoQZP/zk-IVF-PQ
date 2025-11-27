use crate::prelude::*;
use psutil::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub fn get_memory_usage() -> u64 {
    if let Ok(process) = process::Process::new(std::process::id()) {
        if let Ok(memory_info) = process.memory_info() {
            return memory_info.rss();
        }
    }
    0
}

pub fn measure_memory_usage<F, T>(f: F) -> (T, u64)
where
    F: FnOnce() -> T,
{
    let peak_memory = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    // Clone Arc values for the monitoring thread
    let peak_memory_clone = Arc::clone(&peak_memory);
    let running_clone = Arc::clone(&running);

    // Spawn monitoring thread
    let monitor = thread::spawn(move || {
        while running_clone.load(Ordering::SeqCst) {
            let current = get_memory_usage();

            // Update peak if current is higher
            let mut peak = peak_memory_clone.load(Ordering::SeqCst);
            while current > peak {
                match peak_memory_clone.compare_exchange(
                    peak,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }

            thread::sleep(Duration::from_millis(10)); // Sample more frequently
        }
    });

    // Run the actual function
    let result = f();

    // Stop monitoring
    running.store(false, Ordering::SeqCst);
    monitor.join().unwrap();

    // Return peak memory
    let peak = peak_memory.load(Ordering::SeqCst);

    (result, peak)
}

pub fn metrics_eval(
    builder: CircuitBuilder<F, D>,
    pw: PartialWitness<F>,
) -> Result<(f64, f64, f64, u64, u64), Box<dyn std::error::Error>> {
    // 1. 构建电路
    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    let build_time = curr_time.elapsed().as_secs_f64(); // 秒

    // 2. 生成证明
    curr_time = Instant::now();
    let (proof_result, memory_used) = measure_memory_usage(|| data.prove(pw));
    let proof = proof_result?;
    let prove_time = curr_time.elapsed().as_secs_f64(); // 秒

    // 3. 压缩证明并统计大小
    let compressed_proof = data.compress(proof.clone())?;
    let compressed_bytes = compressed_proof.to_bytes();
    let proof_size = compressed_bytes.len(); // 字节数

    // 4. 验证证明
    curr_time = Instant::now();
    data.verify(proof)?;
    let verify_time = curr_time.elapsed().as_secs_f64(); // 秒

    Ok((
        build_time,
        prove_time,
        verify_time,
        proof_size as u64,
        memory_used,
    ))
}
