use crate::hash_gadgets::fs_oracle;
use crate::ivf_flat_verify::gadgets::ivf_flat_verify_gadget;
use crate::prelude::*;
use crate::utils::metrics::metrics_eval;

fn l2_u64(a: u64, b: u64) -> u128 {
    let diff = if a >= b { a - b } else { b - a };
    let diff = diff as u128;
    diff * diff
}

fn distance_u64(src: &[u64], dst: &[u64]) -> u128 {
    src.iter()
        .zip(dst.iter())
        .map(|(&a, &b)| l2_u64(a, b))
        .sum()
}

pub fn ivf_flat_verify_proof(
    ivf_centers: Vec<Vec<u64>>,    // (n_list,d)
    query: Vec<u64>,               // (d,)
    sorted_idx_dis: Vec<Vec<u64>>, // (n_list,2)
    vecss: Vec<Vec<Vec<u64>>>,     // (n_probe,n,d)
    valids: Vec<Vec<u64>>,         // (n_probe,n)
    itemss: Vec<Vec<u64>>,         // (n_probe,n)
    top_k: usize,                  // 明确取哪top_k
) -> Result<(f64, f64, f64, u64, u64, u64), Box<dyn std::error::Error>> {
    let n_list = ivf_centers.len();
    let d = query.len();
    let n_probe = vecss.len();
    let n = vecss[0].len();

    if top_k > n_probe * n {
        return Err(format!("top_k={} > n_probe*n={}", top_k, n_probe * n).into());
    }
    for i in 0..n_probe {
        if vecss[i].len() != n || valids[i].len() != n || itemss[i].len() != n {
            return Err("shape mismatch in vecss/valids/itemss".into());
        }
        for j in 0..n {
            if vecss[i][j].len() != d {
                return Err("shape mismatch in vecss[*][*][d]".into());
            }
        }
    }

    let fs_hash = fs_oracle(query.clone(), 4);

    let max_dis: u64 = (1_u64 << 62) - 1;
    let mut ordered_items_dis: Vec<Vec<u64>> = Vec::with_capacity(n_probe * n);
    for i in 0..n_probe {
        for j in 0..n {
            let mut dis = distance_u64(&query, &vecss[i][j]);
            if valids[i][j] == 0 {
                dis = max_dis as u128;
            }
            if dis > max_dis as u128 {
                return Err(
                    format!("distance overflow: dis={} (max allowed {})", dis, max_dis).into(),
                );
            }
            ordered_items_dis.push(vec![itemss[i][j], dis as u64]);
        }
    }
    ordered_items_dis.sort_by_key(|row| row[1]);

    let mut builder = make_builder();
    let fs_hash_targets = builder.add_virtual_targets(4);
    let ivf_centers_targets = add_targets_2d(&mut builder, vec![n_list, d]);
    let query_targets = builder.add_virtual_targets(d);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![n_list, 2]);
    let vecss_targets = add_targets_3d(&mut builder, vec![n_probe, n, d]);
    let valids_targets = add_targets_2d(&mut builder, vec![n_probe, n]);
    let itemss_targets = add_targets_2d(&mut builder, vec![n_probe, n]);
    let ordered_items_dis_targets = add_targets_2d(&mut builder, vec![n_probe * n, 2]);

    ivf_flat_verify_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        ivf_centers_targets.clone(),
        query_targets.clone(),
        sorted_idx_dis_targets.clone(),
        vecss_targets.clone(),
        valids_targets.clone(),
        itemss_targets.clone(),
        ordered_items_dis_targets.clone(),
        top_k,
    );

    public_targets_1d(&mut builder, query_targets.clone());

    let curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_1d(&mut pw, fs_hash_targets, fs_hash)?;
    input_targets_2d(&mut pw, ivf_centers_targets, ivf_centers)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    input_targets_3d(&mut pw, vecss_targets, vecss)?;
    input_targets_2d(&mut pw, valids_targets, valids)?;
    input_targets_2d(&mut pw, itemss_targets, itemss)?;
    input_targets_2d(&mut pw, ordered_items_dis_targets, ordered_items_dis)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    let metrics = metrics_eval(builder, pw)?;
    Ok(metrics)
}
