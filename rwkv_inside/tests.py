def comprehensive_attention_mimicking_loss(self,teacher_hidden_states, student_hidden_states, layer_idx=0, n_layer=32, args=None):
        batch_size, seq_len, hidden_dim = teacher_hidden_states.shape
        losses = {}
        device = teacher_hidden_states.device
        
        # Dynamically adjusts for output scale discrepancies
        teacher_scale = teacher_hidden_states.detach().abs().mean()
        student_scale = student_hidden_states.detach().abs().mean()
        scale_ratio = teacher_scale / (student_scale + 1e-8)
        
        # Scaled Vector Norm Loss
        scaled_student = student_hidden_states * scale_ratio
        content_loss = torch.linalg.vector_norm(teacher_hidden_states - scaled_student, dim=-1).mean() * (hidden_dim ** -0.5)
        losses['content'] = content_loss
        
        # Cosine Similarity Loss 
        cos_loss = 1 - torch.cosine_similarity(teacher_hidden_states, student_hidden_states, dim=-1).mean()
        losses['cosine'] = cos_loss
        
        # Normalize to make it scale invariant
        t_norm = F.normalize(teacher_hidden_states, dim=-1)
        s_norm = F.normalize(student_hidden_states, dim=-1)
        
        teacher_context = torch.bmm(t_norm, t_norm.transpose(1, 2))
        student_context = torch.bmm(s_norm, s_norm.transpose(1, 2))
        
        # calc context_loss
        context_loss = torch.norm(teacher_context - student_context, p='fro').mean() / (seq_len ** 2)
        losses['context'] = context_loss
        
        # Chunked matching loss
        chunk_size = max(1, hidden_dim // 16)
        local_t = teacher_hidden_states.view(batch_size, seq_len, -1, chunk_size).mean(dim=-2)
        local_s = student_hidden_states.view(batch_size, seq_len, -1, chunk_size).mean(dim=-2)
        local_loss = F.mse_loss(local_t, local_s) * (chunk_size ** 0.5)
        losses['local'] = local_loss
        
        # global matching loss
        global_t = teacher_hidden_states.mean(dim=1)
        global_s = student_hidden_states.mean(dim=1)
        global_loss = F.mse_loss(global_t, global_s) * (hidden_dim ** 0.5)
        losses['global'] = global_loss
        
        # temporal loss (but this is not sure)
        if seq_len > 1:
            # 時間的変化の一致
            teacher_diff = teacher_hidden_states[:, 1:] - teacher_hidden_states[:, :-1]
            student_diff = student_hidden_states[:, 1:] - student_hidden_states[:, :-1]
            temp_loss = 1 - F.cosine_similarity(
                teacher_diff.view(batch_size, -1),
                student_diff.view(batch_size, -1),
                dim=-1
            ).mean()
            losses['temporal'] = temp_loss
        
        # Spectral Loss
        t_flat = teacher_hidden_states.reshape(-1, hidden_dim)
        s_flat = student_hidden_states.reshape(-1, hidden_dim)
        
        # Spectral Loss
        # but maybe its not working
        if t_flat.size(0) >= hidden_dim // 4:
            try:
                # 上位kの特異値を比較
                k = min(8, hidden_dim // 4)
                _, t_s, _ = torch.svd(t_flat.t() @ t_flat, some=True)
                _, s_s, _ = torch.svd(s_flat.t() @ s_flat, some=True)
                
                # 正規化された特異値の分布一致
                spectral_loss = F.mse_loss(t_s[:k]/t_s[0], s_s[:k]/s_s[0])
                losses['spectral'] = spectral_loss
            except:
                # SVDが収束しない場合は代替手段
                t_cov = (t_flat.t() @ t_flat) / t_flat.size(0)
                s_cov = (s_flat.t() @ s_flat) / s_flat.size(0)
                spectral_loss = torch.norm(t_cov - s_cov, p='fro') / hidden_dim
                losses['spectral'] = spectral_loss
        
        
        # === 9. 層依存パラメータの設定 ===
        # 下位層は基本情報、上位層は高次特徴を重視
        #total_layers = getattr(args, 'total_layers', 24)  # デフォルト値
        relative_depth = layer_idx / n_layer
        
        # 層の深さに応じた重み付け
        layer_weight = 1.0#args.base_weight * (args.layer_decay ** layer_idx)
        
        # 上部層はコンテキスト関係をより重視
        context_importance = 1.0 * (1.0 + 0.5 * relative_depth)
        
        # 下部層は基本情報をより重視
        content_importance = 1.0 * (1.0 - 0.3 * relative_depth)
        
        # === 10. 最終的な重み付けLoss ===
        # 基本Lossの組み合わせ
        combined_loss = (
            content_importance * losses['content'] +
            2.0 * losses['cosine'] +
            context_importance * losses['context'] +
            getattr(args, 'local_weight', 0.5) * losses.get('local', 0) +
            getattr(args, 'global_weight', 0.5) * losses.get('global', 0) +
            getattr(args, 'temporal_weight', 1.0) * losses.get('temporal', 0) +
            getattr(args, 'spectral_weight', 0.3) * losses.get('spectral', 0)
        )
        
        # オプション: 検証用にすべての損失コンポーネントを返す
        if getattr(args, 'return_components', False):
            return layer_weight * combined_loss, losses
        
        return layer_weight * combined_loss