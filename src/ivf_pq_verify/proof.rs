// TODO: 当前先把它写成一个stand alone的版本, 后面再做修改
use crate::hash_gadgets::fs_oracle;
use crate::ivf_pq_verify::gadgets::ivf_pq_verify_gadget;
use crate::prelude::*;

pub fn compress_i64(set: Vec<i64>, alpha: i64) -> u64 {
    let alpha_f = F::from_noncanonical_i64(alpha);
    let mut cur_f = F::from_noncanonical_i64(set[0]);
    for i in 1..set.len() {
        cur_f = cur_f * alpha_f + F::from_noncanonical_i64(set[i]);
    }
    cur_f.to_canonical_u64()
}

pub fn convert_ft_set_i64(
    f_set: Vec<Vec<i64>>,
    t_set: Vec<Vec<i64>>,
    alpha: i64,
) -> (Vec<u64>, Vec<u64>) {
    let mut f: Vec<u64> = f_set
        .into_iter()
        .map(|row| compress_i64(row, alpha))
        .collect();
    let mut t: Vec<u64> = t_set
        .into_iter()
        .map(|row| compress_i64(row, alpha))
        .collect();

    // 扩增到同样长度, 都只用0号位扩增
    let sz = if f.len() > t.len() { f.len() } else { t.len() };
    while t.len() < sz {
        t.push(t[0]);
    }
    while f.len() < sz {
        f.push(f[0]);
    }

    f.sort();
    // t.sort();
    for i in 0..sz {
        if t[i] == f[0] {
            t[i] = t[0];
            t[0] = f[0];
        }
    }
    t[1..].sort();

    // 按规则调整t的位置
    for i in 1..sz {
        if f[i] != f[i - 1] {
            for j in 0..sz {
                if t[j] == f[i] {
                    t[j] = t[i];
                    t[i] = f[i];
                    break;
                }
            }
        }
    }

    (f, t)
}

// NOTE: 使用呢i64防止模回绕
pub fn luts_gen_i64(
    codebooks: &[Vec<Vec<i64>>],
    query: &[i64],
    centers: &[Vec<i64>],
) -> Vec<Vec<Vec<i64>>> {
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let n_probe = centers.len();
    let D_ = query.len(); // M*d=D_

    // 计算挪动后的向量, 注意用i128防回绕
    let mut moved_query: Vec<Vec<i64>> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        let mut row: Vec<i64> = Vec::with_capacity(D_);
        for j in 0..D_ {
            row.push(query[j] - centers[i][j]);
        }
        moved_query.push(row);
    }

    // (n_probe,M,K)
    let mut luts: Vec<Vec<Vec<i64>>> = Vec::with_capacity(n_probe);
    for i in 0..n_probe {
        let curr_query = moved_query[i].clone();
        let mut curr_cube: Vec<Vec<i64>> = Vec::with_capacity(M);
        for j in 0..M {
            let mut curr_row: Vec<i64> = Vec::with_capacity(K);
            for k in 0..K {
                let slide = codebooks[j][k].clone();
                let query_slide = curr_query[j * d..(j + 1) * d].to_vec();
                let mut dis: i64 = 0;
                for t in 0..d {
                    dis += (slide[t] - query_slide[t]).pow(2);
                }
                curr_row.push(dis);
            }
            curr_cube.push(curr_row);
        }
        luts.push(curr_cube);
    }

    luts
}

pub fn dis_cal(dises: Vec<Vec<i64>>) -> Vec<i64> {
    dises.into_iter().map(|row| row.iter().sum()).collect()
}

pub fn ivf_pq_verify_proof(
    ivf_centers: Vec<Vec<i64>>,      // (n_list,D)
    query: Vec<i64>,                 // (D,)
    sorted_idx_dis: Vec<Vec<i64>>,   // (n_list,2)
    filtered_centers: Vec<Vec<i64>>, // (n_probe,D)
    probe_count: Vec<i64>,           // (n_probe,)
    filtered_vecs: Vec<Vec<i64>>,    // (max_sz,M)
    vecs_cluster_hot: Vec<Vec<i64>>, // (max_sz,n_probe)
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
    // let fs_hash = fs_oracle(query.clone(), 5);
    let fs_hash = fs_oracle(
        query.clone().into_iter().map(|item| item as u64).collect(),
        5,
    );
    // let fs_hash: Vec<i64> = vec![1, 2, 3, 4, 5];

    // 计算LUTs
    let centers: Vec<Vec<i64>> = (0..n_probe)
        .map(|i| ivf_centers[sorted_idx_dis[i][0] as usize].clone())
        .collect();
    let luts = luts_gen_i64(&codebooks, &query, &centers);

    // 手算filtered_dis
    let mut filtered_dis: Vec<Vec<i64>> = Vec::with_capacity(max_sz);
    let mut index_list: Vec<i64> = Vec::with_capacity(max_sz);
    for i in 0..max_sz {
        let mut curr_idx: usize = 0;
        for j in 0..n_probe {
            curr_idx += (vecs_cluster_hot[i][j] as usize) * j;
        }
        index_list.push(curr_idx as i64);
        let mut curr_row: Vec<i64> = Vec::with_capacity(M);
        for j in 0..M {
            curr_row.push(luts[curr_idx][j][filtered_vecs[i][j] as usize]);
        }
        filtered_dis.push(curr_row);
    }
    // println!("{:?}", dis_cal(filtered_dis.clone()));

    // 手压大表并生成f_,t_
    let mut lut_set: Vec<Vec<i64>> = Vec::with_capacity(n_probe * M * K);
    for i in 0..n_probe {
        for j in 0..M {
            for k in 0..K {
                lut_set.push(vec![i as i64, j as i64, k as i64, luts[i][j][k]]);
            }
        }
    }
    let mut dis_set: Vec<Vec<i64>> = Vec::with_capacity(max_sz * M);
    for i in 0..max_sz {
        for j in 0..M {
            dis_set.push(vec![
                index_list[i],
                j as i64,
                filtered_vecs[i][j],
                filtered_dis[i][j],
            ]);
        }
    }
    let (f_, t_) = convert_ft_set_i64(dis_set, lut_set, fs_hash[2] as i64);
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
    input_targets_2d_sign(&mut pw, ivf_centers_targets, ivf_centers)?;
    input_targets_1d_sign(&mut pw, query_targets, query)?;
    input_targets_2d_sign(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    input_targets_2d_sign(&mut pw, filtered_centers_targets, filtered_centers)?;
    input_targets_1d_sign(&mut pw, probe_count_targets, probe_count)?;
    input_targets_2d_sign(&mut pw, filtered_vecs_targets, filtered_vecs)?;
    input_targets_2d_sign(&mut pw, filtered_dis_targets, filtered_dis)?;
    input_targets_2d_sign(&mut pw, vecs_cluster_hot_targets, vecs_cluster_hot)?;
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
