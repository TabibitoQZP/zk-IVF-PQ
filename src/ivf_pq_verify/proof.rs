use crate::hash_gadgets::fs_oracle;
use crate::ivf_pq_verify::gadgets::ivf_pq_verify_gadget;
use crate::pq_flat_verify::proof::convert_ft_set;
use crate::prelude::*;

// NOTE: 如果不发生模回绕, 那计算就无所谓
pub fn luts_gen_u64(
    codebooks: &[Vec<Vec<i64>>],
    query: &[u64],
    centers: &[Vec<u64>],
) -> Vec<Vec<Vec<u64>>> {
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let n_probe = centers.len();
    let D_ = query.len(); // M*d=D_

    // 计算挪动后的向量, 注意用i128防回绕
    let mut moved_query: Vec<Vec<i128>> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        let mut row: Vec<i128> = Vec::with_capacity(D_);
        for j in 0..D_ {
            row.push(query[j] as i128 - centers[i][j] as i128);
        }
        moved_query.push(row);
    }

    // (n_probe,M,K)
    let mut luts: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        let curr_query = moved_query[i].clone();
        let mut curr_cube: Vec<Vec<u64>> = Vec::with_capacity(M);
        for j in 0..M {
            let mut curr_row: Vec<u64> = Vec::with_capacity(K);
            for k in 0..K {
                let slide = codebooks[j][k].clone();
                let query_slide = curr_query[j * d..(j + 1) * d].to_vec();
                let mut dis: i128 = 0;
                for t in 0..d {
                    dis += (slide[t] as i128 - query_slide[t]).pow(2);
                }
                curr_row.push(dis as u64);
            }
            curr_cube.push(curr_row);
        }
        luts.push(curr_cube);
    }

    luts
}

pub fn ivf_pq_verify_proof(
    ivf_centers: Vec<Vec<u64>>,      // (n_list,D)
    query: Vec<u64>,                 // (D,)
    sorted_idx_dis: Vec<Vec<u64>>,   // (n_list,2)
    filtered_centers: Vec<Vec<u64>>, // (n_probe,D)
    probe_count: Vec<u64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<u64>>,    // (max_sz,M)
    vecs_cluster_hot: Vec<Vec<u64>>, // (max_sz,n_probe)
    codebooks: Vec<Vec<Vec<i64>>>,   // (M,K,d)
) -> Result<(), Box<dyn std::error::Error>> {
    // 初始化维度信息
    let n_list = ivf_centers.len();
    let D_ = query.len();
    let n_probe = probe_count.len();
    let max_sz = filtered_vecs.len();
    let M = filtered_vecs[0].len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();

    // 初始化F-S哈希值
    let fs_hash = fs_oracle(query.clone(), 5);

    // 计算LUTs
    let centers: Vec<Vec<u64>> = (0..n_probe)
        .map(|i| ivf_centers[sorted_idx_dis[i][0] as usize].clone())
        .collect();
    let luts = luts_gen_u64(&codebooks, &query, &centers);

    // 手算filtered_dis
    let mut filtered_dis: Vec<Vec<u64>> = Vec::with_capacity(max_sz);
    let mut index_list: Vec<u64> = Vec::with_capacity(max_sz);
    for i in 0..max_sz {
        let mut curr_idx: usize = 0;
        for j in 0..n_probe {
            curr_idx += (vecs_cluster_hot[i][j] as usize) * j;
        }
        index_list.push(curr_idx as u64);
        let mut curr_row: Vec<u64> = Vec::with_capacity(M);
        for j in 0..M {
            curr_row.push(luts[curr_idx][j][filtered_vecs[i][j] as usize]);
        }
        filtered_dis.push(curr_row);
    }

    // 手压大表并生成f_,t_
    let mut lut_set: Vec<Vec<u64>> = Vec::with_capacity(n_probe * M * K);
    for i in 0..n_probe {
        for j in 0..M {
            for k in 0..K {
                lut_set.push(vec![i as u64, j as u64, k as u64, luts[i][j][k]]);
            }
        }
    }
    let mut dis_set: Vec<Vec<u64>> = Vec::with_capacity(max_sz * M);
    for i in 0..max_sz {
        for j in 0..M {
            dis_set.push(vec![
                index_list[i],
                j as u64,
                filtered_vecs[i][j],
                filtered_dis[i][j],
            ]);
        }
    }
    let (f_, t_) = convert_ft_set(dis_set, lut_set, fs_hash[2]);
    let f_t_sz = f_.len();

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let fs_hash_targets = builder.add_virtual_targets(5);
    let ivf_centers_targets = add_targets_2d(&mut builder, vec![n_list, D_]);
    let query_targets = builder.add_virtual_targets(D_);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![n_list, 2]);
    let filtered_centers_targets = add_targets_2d(&mut builder, vec![n_probe, D_]);
    let probe_count_targets = builder.add_virtual_targets(n_probe);
    let filtered_vecs_targets = add_targets_2d(&mut builder, vec![max_sz, M]);
    let filtered_dis_targets = add_targets_2d(&mut builder, vec![max_sz, M]);
    let vecs_cluster_hot_targets = add_targets_2d(&mut builder, vec![max_sz, n_probe]);
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);
    let f__targets = builder.add_virtual_targets(f_t_sz);
    let t__targets = builder.add_virtual_targets(f_t_sz);

    // 构建电路
    ivf_pq_verify_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        ivf_centers_targets.clone(),
        query_targets.clone(),
        sorted_idx_dis_targets.clone(),
        filtered_centers_targets.clone(),
        probe_count_targets.clone(),
        filtered_vecs_targets.clone(),
        filtered_dis_targets.clone(),
        vecs_cluster_hot_targets.clone(),
        codebooks_targets.clone(),
        f__targets.clone(),
        t__targets.clone(),
    );

    // 设置公开输入和witness
    public_targets_1d(&mut builder, query_targets.clone());

    // 构建电路
    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    println!("构建电路耗时: {:?}", curr_time.elapsed());

    // 输入公开输入和witness
    curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_1d(&mut pw, fs_hash_targets, fs_hash)?;
    input_targets_2d(&mut pw, ivf_centers_targets, ivf_centers)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    input_targets_2d(&mut pw, filtered_centers_targets, filtered_centers)?;
    input_targets_1d(&mut pw, probe_count_targets, probe_count)?;
    input_targets_2d(&mut pw, filtered_vecs_targets, filtered_vecs)?;
    input_targets_2d(&mut pw, filtered_dis_targets, filtered_dis)?;
    input_targets_2d(&mut pw, vecs_cluster_hot_targets, vecs_cluster_hot)?;
    input_targets_3d_sign(&mut pw, codebooks_targets, codebooks)?;
    input_targets_1d(&mut pw, f__targets, f_)?;
    input_targets_1d(&mut pw, t__targets, t_)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    // 证明生成和验证
    curr_time = Instant::now();
    let proof = data.prove(pw)?;
    println!("证明生成: {:?}", curr_time.elapsed());

    // 证明大小
    let compressed_proof = data.compress(proof.clone())?;
    let compressed_bytes = compressed_proof.to_bytes();
    println!("证明大小: {}B", compressed_bytes.len());

    curr_time = Instant::now();
    let _ = data.verify(proof);
    println!("证明验证: {:?}", curr_time.elapsed());
    Ok(())
}
