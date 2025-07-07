def compute_kl_loss(student_outputs, teacher_logits, labels, args, attention_mask=None, chunk_size=4096,temperature=1.0):
    student_logits = student_outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    vocab_student = student_logits.shape[-1]
    vocab_teacher = teacher_logits.shape[-1]

    # Truncate teacher logits if necessary
    if vocab_teacher > vocab_student:
        teacher_logits = teacher_logits[:, :, :vocab_student]
    
    # 温度パラメータの取得（argsにtemperature属性があると仮定）
    #temperature = getattr(args, "temperature", 2.0)

      
    # 温度スケーリングを適用したロジットの計算
    student_logits_scaled = student_logits / temperature
    
    # Compute softmax for student and teacher with temperature scaling
    log_probs_student = F.log_softmax(student_logits_scaled, dim=-1)  # [batch_size, seq_len, vocab_size]
    with torch.no_grad():
        teacher_logits_scaled = teacher_logits / temperature
        targets = F.softmax(teacher_logits_scaled, dim=-1)    # [batch_size, seq_len, vocab_size]
    
    # Compute KL divergence without reduction
    kl_div_all = F.kl_div(
        log_probs_student,
        targets,
        reduction='none'  # Keep the full tensor to apply mask
    )  # [batch_size, seq_len, vocab_size]
    
    # Sum across vocabulary dimension first
    kl_div_per_token = kl_div_all.sum(dim=-1)  # [batch_size, seq_len]
    
    if attention_mask is not None:
        # Apply attention mask and compute mean only over attended positions
        masked_kl = kl_div_per_token * attention_mask
        kl_loss = masked_kl.sum() / (attention_mask.sum() + 1e-6)  # Add small epsilon for numerical stability
    else:
        # If no mask provided, take mean over all tokens
        kl_loss = kl_div_per_token.mean()
    
    # 温度スケーリングによる勾配補正
    kl_loss = (temperature ** 2) * kl_loss

    del log_probs_student, targets, kl_div_all, kl_div_per_token
    
    # Get cross entropy loss from student outputs
    student_cross_entropy_loss = student_outputs.loss
    
    # Combine losses using weights from args
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    
    del student_logits, teacher_logits, labels
    if attention_mask is not None:
        del attention_mask
    return loss, kl_loss, student_cross_entropy_loss