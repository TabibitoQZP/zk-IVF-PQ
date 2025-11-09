use crate::hash_gadgets::fs_oracle;
use crate::hash_gadgets::hash_u64;
use crate::pq_flat_com::gadgets::pq_flat_com_gadget;
use crate::prelude::*;

// 计算距离
pub fn dis_u64(a: Vec<u64>, b: Vec<u64>) -> u64 {
    let mut result = 0;
    let sz = a.len();
    for i in 0..sz {
        let mut delta: u64 = 0;
        if a[i] > b[i] {
            delta = a[i] - b[i];
        } else {
            delta = b[i] - a[i];
        }
        result += delta * delta;
    }
    result
}

// 打表
pub fn lut_gen_u64(codebooks: &[Vec<Vec<u64>>], query: &[u64]) -> Vec<Vec<u64>> {
    let M = codebooks.len();
    let K = codebooks[0].len();
    let d = codebooks[0][0].len();
    let mut lut: Vec<Vec<u64>> = Vec::with_capacity(M);
    for i in 0..M {
        let mut cur_result: Vec<u64> = Vec::with_capacity(K);
        for j in 0..K {
            let cur_vec = codebooks[i][j].clone(); // (d,)
            let query_slide = query[(i * d)..((i + 1) * d)].to_vec();
            let dis = dis_u64(cur_vec, query_slide);
            cur_result.push(dis);
        }
        lut.push(cur_result);
    }
    lut
}

// 计算log2
pub fn log2(mut n: u64) -> u64 {
    let mut cnt: u64 = 0;
    while n > 1 {
        n /= 2;
        cnt += 1;
    }
    cnt
}

pub fn level_hash(hash_list: Vec<u64>) -> Vec<u64> {
    let sz = hash_list.len() / 2;
    let mut result: Vec<u64> = Vec::with_capacity(sz);
    for i in 0..sz {
        result.push(hash_u64(vec![hash_list[2 * i], hash_list[2 * i + 1]]));
    }
    result
}

// 将leaves的merkle哈希转成二叉树
pub fn merkle_tree_u64(leaves: Vec<Vec<u64>>) -> Vec<u64> {
    let mut sz = leaves.len(); // 只考虑sz=2^k的情况
    let mut result: Vec<u64> = Vec::with_capacity(2 * sz - 1);
    let k = log2(sz as u64);

    // 将leaves哈希化
    let mut hash_leaves: Vec<u64> = Vec::with_capacity(sz);
    for item in leaves {
        hash_leaves.push(hash_u64(item));
    }

    // 获得层级哈希
    let mut level_tree: Vec<Vec<u64>> = Vec::with_capacity(k as usize + 1);
    while sz > 0 {
        level_tree.push(hash_leaves.clone());
        hash_leaves = level_hash(hash_leaves);
        sz /= 2;
    }
    for item in level_tree.into_iter().rev() {
        result.extend(item);
    }
    result
}

// 将LUT表上承诺, 并返回一个二叉树 (列表存储)
pub fn merkle_lut2d_u64(lut: &[Vec<u64>]) -> Vec<u64> {
    let x = lut.len();
    let y = lut[0].len();

    let mut leaves: Vec<Vec<u64>> = Vec::with_capacity(x);
    for i in 0..x {
        for j in 0..y {
            leaves.push(vec![i as u64, j as u64, lut[i][j]]);
        }
    }
    merkle_tree_u64(leaves)
}

// 将完全二叉树叶子从左到右视为[0,2^depth)的列表, 计算到达路径
pub fn binanry_decompose(mut x: u64, step: u64) -> Vec<u64> {
    let mut result: Vec<u64> = Vec::new();
    for _ in 0..step {
        result.push(x - x / 2 * 2);
        x /= 2;
    }
    result.reverse();
    // 从根结点开始, 0代表向左, 1代表向右
    result
}

// 计算lut的表格
pub fn lut2d_path_u64(shape: &[u64], lut_tree: &[u64], i: u64, j: u64) -> Vec<Vec<u64>> {
    let sz = shape[0] * shape[1];
    let step = log2(sz);
    let idx = shape[1] * i + j; // 叶子结点的索引
    let path = binanry_decompose(idx, step);
    let mut curr_idx: u64 = 0; // 当前处在root

    // 计算merkle的迹
    let mut merkle_trace: Vec<Vec<u64>> = Vec::with_capacity(step as usize);
    for direct in path {
        if direct == 0 {
            curr_idx = curr_idx * 2 + 1;
            merkle_trace.push(vec![direct, lut_tree[(curr_idx + 1) as usize]]);
        } else {
            curr_idx = curr_idx * 2 + 2;
            merkle_trace.push(vec![direct, lut_tree[(curr_idx - 1) as usize]]);
        }
    }
    // 记得反转, 变成自底向上计算
    merkle_trace.reverse();
    merkle_trace
}

pub fn pq_flat_com_proof(
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
    let shape = vec![M as u64, K as u64];

    let fs_hash = fs_oracle(query.clone(), 2);
    let lut = lut_gen_u64(&codebooks, &query);
    let lut_tree = merkle_lut2d_u64(&lut);
    // 这里我们手写一个pq_dis
    let mut pq_dis: Vec<Vec<u64>> = Vec::with_capacity(N);
    // 手算merkle_path
    let mut merkle_path: Vec<Vec<Vec<Vec<u64>>>> = Vec::new();
    for i in 0..N {
        let mut curr_dis: Vec<u64> = Vec::with_capacity(M);
        let mut curr_merkle: Vec<Vec<Vec<u64>>> = Vec::with_capacity(M);
        for j in 0..M {
            curr_dis.push(lut[j][pq_vecs[i][j] as usize]);
            curr_merkle.push(lut2d_path_u64(&shape, &lut_tree, j as u64, pq_vecs[i][j]));
        }
        pq_dis.push(curr_dis);
        merkle_path.push(curr_merkle);
    }

    let step = merkle_path[0][0].len();

    // 初始化电路
    let mut builder = make_builder();

    // 初始化输入
    let fs_hash_targets = builder.add_virtual_targets(2);
    let codebooks_targets = add_targets_3d(&mut builder, vec![M, K, d]);
    let query_targets = builder.add_virtual_targets(D_);
    let pq_vecs_targets = add_targets_2d(&mut builder, vec![N, M]);
    let pq_dis_targets = add_targets_2d(&mut builder, vec![N, M]);
    let merkle_path_targets = add_targets_4d(&mut builder, vec![N, M, step, 2]);
    let sorted_idx_dis_targets = add_targets_2d(&mut builder, vec![N, 2]);

    // 构建电路
    pq_flat_com_gadget(
        &mut builder,
        fs_hash_targets.clone(),
        codebooks_targets.clone(),
        query_targets.clone(),
        pq_vecs_targets.clone(),
        pq_dis_targets.clone(),
        merkle_path_targets.clone(),
        sorted_idx_dis_targets.clone(),
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
    input_targets_3d(&mut pw, codebooks_targets, codebooks)?;
    input_targets_1d(&mut pw, query_targets, query)?;
    input_targets_2d(&mut pw, pq_vecs_targets, pq_vecs)?;
    input_targets_2d(&mut pw, pq_dis_targets, pq_dis)?;
    input_targets_4d(&mut pw, merkle_path_targets, merkle_path)?;
    input_targets_2d(&mut pw, sorted_idx_dis_targets, sorted_idx_dis)?;
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
