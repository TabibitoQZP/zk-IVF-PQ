use crate::hash_gadgets::fs_oracle;
use crate::pq_flat_com::proof::lut_gen_u64;
use crate::pq_flat_verify::gadgets::pq_flat_verify_gadget;
use crate::prelude::*;

pub fn compress_u64(set: Vec<u64>, alpha: u64) -> u64 {
    let alpha_f = F::from_canonical_u64(alpha);
    let mut cur_f = F::from_canonical_u64(set[0]);
    for i in 1..set.len() {
        cur_f = cur_f * alpha_f + F::from_canonical_u64(set[i]);
    }
    cur_f.to_canonical_u64()
}

pub fn convert_ft_set(
    f_set: Vec<Vec<u64>>,
    t_set: Vec<Vec<u64>>,
    alpha: u64,
) -> (Vec<u64>, Vec<u64>) {
    let mut f: Vec<u64> = f_set
        .into_iter()
        .map(|row| compress_u64(row, alpha))
        .collect();
    let mut t: Vec<u64> = t_set
        .into_iter()
        .map(|row| compress_u64(row, alpha))
        .collect();

    // 扩增到同样长度, 都只用0号位扩增
    let sz = if f.len() > t.len() { f.len() } else { t.len() };
    while t.len() < sz {
        t.push(t[0]);
    }
    while f.len() < sz {
        f.push(f[0]);
    }

    // 对t也排序, 这样第一个必然相同
    f.sort();
    t.sort();

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

pub fn pq_flat_verify_proof(
    codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
    query: Vec<u64>,               // (D,)
    pq_vecs: Vec<Vec<u64>>,        // (N,M)
    sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
) -> Result<(), Box<dyn std::error::Error>> {
    let D_ = query.len();
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let N = pq_vecs.len();

    // 初始化F-S哈希值
    let fs_hash = fs_oracle(query.clone(), 5);

    // # 手算f_, t_对应的值
    // 1. 计算LUT
    let lut = lut_gen_u64(&codebooks, &query);
    // 2. 手算pq_dis列表
    let pq_dis: Vec<Vec<u64>> = (0..N)
        .map(|i| (0..M).map(|j| lut[j][pq_vecs[i][j] as usize]).collect())
        .collect();
    // 3. 计算f_和s_
    let mut f_set: Vec<Vec<u64>> = Vec::with_capacity(N * M);
    let mut t_set: Vec<Vec<u64>> = Vec::with_capacity(N * M);
    for i in 0..N {
        for j in 0..M {
            f_set.push(vec![j as u64, pq_vecs[i][j], pq_dis[i][j]]);
        }
    }
    for i in 0..M {
        for j in 0..K {
            t_set.push(vec![i as u64, j as u64, lut[i][j]]);
        }
    }
    let (f_, t_) = convert_ft_set(f_set, t_set, fs_hash[2]);
    // println!("{} {} {}", f_.len(), N, M);

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let fs_hash_targets = builder.add_virtual_targets(5);
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);
    let query_targets = builder.add_virtual_targets(D_);
    let pq_vecs_targets = add_targets_2d(&mut builder, vec![N, M]);
    let pq_dis_targets = add_targets_2d(&mut builder, vec![N, M]);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![N, 2]);
    let f__targets = builder.add_virtual_targets(N * M);
    let t__targets = builder.add_virtual_targets(N * M);

    pq_flat_verify_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        codebooks_targets.clone(),
        query_targets.clone(),
        pq_vecs_targets.clone(),
        pq_dis_targets.clone(),
        sorted_idx_dis_targets.clone(),
        f__targets.clone(),
        t__targets.clone(),
    );

    // 设置公开输入和witness
    public_targets_1d(&mut builder, query_targets.clone());
    public_targets_1d(&mut builder, fs_hash_targets.clone());

    // 构建电路
    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    println!("构建电路耗时: {:?}", curr_time.elapsed());

    // 输入公开输入和witness
    curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    input_targets_1d(&mut pw, fs_hash_targets, fs_hash)?;
    input_targets_3d(&mut pw, codebooks_targets, codebooks)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, pq_vecs_targets, pq_vecs)?;
    input_targets_2d(&mut pw, pq_dis_targets, pq_dis)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
    input_targets_1d(&mut pw, f__targets, f_)?;
    input_targets_1d(&mut pw, t__targets, t_)?;
    println!("输入witness: {:?}", curr_time.elapsed());

    // 证明生成和验证
    curr_time = Instant::now();
    let proof = data.prove(pw)?;
    println!("证明生成: {:?}", curr_time.elapsed());
    curr_time = Instant::now();
    let _ = data.verify(proof);
    println!("证明验证: {:?}", curr_time.elapsed());
    Ok(())
}
