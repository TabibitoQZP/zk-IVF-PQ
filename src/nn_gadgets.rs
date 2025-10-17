use crate::hash_gadgets::*;
use crate::prelude::*;

pub fn l2(builder: &mut CircuitBuilder<F, D>, a: Target, b: Target) -> Target {
    let delta = builder.sub(a, b);
    builder.square(delta)
}

pub fn distance(builder: &mut CircuitBuilder<F, D>, src: Vec<Target>, dst: Vec<Target>) -> Target {
    let d = src.len();
    let mut cur_target = l2(builder, src[0], dst[0]);
    for i in 1..d {
        let tmp_target = l2(builder, src[i], dst[i]);
        cur_target = builder.add(cur_target, tmp_target);
    }
    cur_target
}

// 这个电路证明了最近邻nn
pub fn nn_gadget(
    builder: &mut CircuitBuilder<F, D>,
    r: Target,
    t: Target,
    src_vecs: Vec<Vec<Target>>,
    query: Vec<Target>,
    sorted_idx_dis: Vec<Vec<Target>>,
) -> Target {
    let n = src_vecs.len();
    let d = src_vecs[0].len();

    // 将索引码也嵌入
    let mut src_with_idx: Vec<Vec<Target>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut tmp_vec: Vec<Target> = Vec::with_capacity(d + 1);
        tmp_vec.push(builder.constant(F::from_canonical_u64(i as u64)));
        tmp_vec.extend(src_vecs[i].clone());
        src_with_idx.push(tmp_vec);
    }

    // 计算merkle根
    let root = merkle_tree_gadget(builder, src_with_idx);

    // 计算未排序的(idx, dis) 对
    let mut unsorted_idx_dis: Vec<Vec<Target>> = Vec::with_capacity(n);
    for i in 0..n {
        unsorted_idx_dis.push(vec![
            builder.constant(F::from_canonical_u64(i as u64)),
            distance(builder, query.clone(), src_vecs[i].clone()),
        ])
    }
    set_equal_gadget(builder, r, t, unsorted_idx_dis, sorted_idx_dis.clone());

    // 验证排序是否是递增序
    for i in 0..n - 1 {
        let tmp_target = builder.sub(
            sorted_idx_dis[i + 1][1].clone(),
            sorted_idx_dis[i][1].clone(),
        );
        builder.range_check(tmp_target, 32);
    }
    root
}

pub fn nn_prove(
    src_vecs: Vec<Vec<u64>>,
    query: Vec<u64>,
    root: u64,
    sorted_idx_dis: Vec<Vec<u64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = src_vecs.len();
    let d = src_vecs[0].len();
    let mut builder = make_builder();

    // F-S过程获取r和t
    let r = hash_u64(query.clone());
    let mut tmp_vec: Vec<u64> = Vec::with_capacity(query.len() + 1);
    tmp_vec.push(r);
    tmp_vec.extend(query.clone());
    let t = hash_u64(tmp_vec);

    // 设置电路接口
    let query_target = builder.add_virtual_targets(d);
    for i in 0..d {
        builder.register_public_input(query_target[i]);
    }
    let mut src_vecs_target: Vec<Vec<Target>> = Vec::with_capacity(n);
    for _ in 0..n {
        src_vecs_target.push(builder.add_virtual_targets(d));
    }
    let r_target = builder.add_virtual_target();
    let t_target = builder.add_virtual_target();
    let mut sorted_idx_dis_target: Vec<Vec<Target>> = Vec::with_capacity(n);
    for _ in 0..n {
        sorted_idx_dis_target.push(builder.add_virtual_targets(2));
    }

    // 设置电路连线并获取承诺根
    let root_target = nn_gadget(
        &mut builder,
        r_target,
        t_target,
        src_vecs_target.clone(),
        query_target.clone(),
        sorted_idx_dis_target.clone(),
    );
    builder.register_public_input(root_target);
    builder.register_public_input(r_target);
    builder.register_public_input(t_target);

    // 构建电路
    let mut curr_time = Instant::now();
    let data = builder.build::<C>();
    println!("构建电路耗时: {:?}", curr_time.elapsed());

    // 输入witness
    curr_time = Instant::now();
    let mut pw = PartialWitness::new();
    for i in 0..n {
        for j in 0..d {
            pw.set_target(src_vecs_target[i][j], F::from_canonical_u64(src_vecs[i][j]))?;
        }
    }
    for i in 0..d {
        pw.set_target(query_target[i], F::from_canonical_u64(query[i]))?;
    }
    for i in 0..n {
        pw.set_target(
            sorted_idx_dis_target[i][0],
            F::from_canonical_u64(sorted_idx_dis[i][0]),
        )?;
        pw.set_target(
            sorted_idx_dis_target[i][1],
            F::from_canonical_u64(sorted_idx_dis[i][1]),
        )?;
    }
    pw.set_target(r_target, F::from_canonical_u64(r))?;
    pw.set_target(t_target, F::from_canonical_u64(t))?;
    pw.set_target(root_target, F::from_canonical_u64(root))?;
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
