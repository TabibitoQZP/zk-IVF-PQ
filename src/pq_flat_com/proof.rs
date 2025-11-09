use crate::hash_gadgets::fs_oracle;
use crate::pq_flat_com::gadgets::pq_flat_com_gadget;
use crate::prelude::*;

// pub fn pq_flat_proof(
//     codebooks: Vec<Vec<Vec<u64>>>, // (M,K,d)
//     query: Vec<u64>,               // (D,)
//     pq_vecs: Vec<Vec<u64>>,        // (N,M)
//     pq_dis: Vec<Vec<u64>>,         // (N,M)
//     sorted_idx_dis: Vec<Vec<u64>>, // (N,2)
// ) -> Result<(), Box<dyn std::error::Error>> {
//     // TODO: 基于pq_vecs打表
// }
