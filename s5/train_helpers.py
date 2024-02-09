from functools import partial
import jax
import jax.numpy as np
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any, Tuple
from optax import ctc_loss
import numpy as onp
import torch
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from .dataloading import ALPHABET

# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val, dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       dt_global=False
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
            integration_timesteps = np.ones((2*bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        dummy_input = np.ones((bsz, seq_len, in_dim))
        integration_timesteps = np.ones((bsz, seq_len, ))
        day_idxs = np.zeros((bsz,)).astype(int)

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           dummy_input, integration_timesteps, day_idxs,
                           )
    if batchnorm:
        params = variables["params"].unfreeze()
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"].unfreeze()
        # Note unfreeze() is for using Optax.

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Train and eval steps
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label

beam_search_decoder = ctc_decoder(
    lexicon=None,
    tokens=ALPHABET,
    lm=None,
    nbest=1,
    beam_size=50,
    sil_token=' ',
)

def compute_ctc_accuracy(logits, label, neural_padding, label_padding):
    # convert to torch for CTC decode & accuracy
    logits = onp.array(logits)
    neural_padding = onp.array(neural_padding)
    label_padding = onp.array(label_padding)
    label = onp.array(label)
    logits_torch = torch.from_numpy(onp.array(logits[None,neural_padding==0,:]))
    beam_search_result = beam_search_decoder(logits_torch)
    tokens = beam_search_result[0][0].tokens
    predict = [ALPHABET[token] for token in tokens]
#     print('predict')
#     print(predict)
    actual_label = label[label_padding==0].astype(int)
    actual_phonemes = [ALPHABET[token] for token in actual_label]
#     print('actual_phonemes')
#     print(actual_phonemes)
    # outputs an array with edit distance and length
    edit_distance = torchaudio.functional.edit_distance(actual_phonemes, predict)
#     print('edit_distance')
#     print(edit_distance)
    length = len(actual_phonemes)
    return [edit_distance, length]


def prep_batch(batch: tuple,
               seq_len: int,
               in_dim: int) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """

    inputs, targets, neural_pad, sentence_pad, day_idxs = batch

    # Convert to JAX.
    inputs = np.asarray(inputs.numpy())

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = np.pad(inputs, ((0, 0), (0, num_pad)), 'constant', constant_values=(0,))
        raise RuntimeError("BCI: We should not be getting into this situation.")

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)
        raise RuntimeError("BCI: We should not be getting into this situation.")

    # Convert and apply.
    targets = np.array(targets.numpy()).astype(float)
    neural_pad = np.array(neural_pad.numpy()).astype(float)
    sentence_pad = np.array(sentence_pad.numpy()).astype(float)
    day_idxs = np.array(day_idxs.numpy()).astype(int)
    
    # If there is an aux channel containing the integration times, then add that.
    # if 'timesteps' in aux_data.keys():
    #     integration_timesteps = np.diff(np.asarray(aux_data['timesteps'].numpy()))
    # else:
    integration_timesteps = np.ones((len(inputs), seq_len))

    return inputs, targets, integration_timesteps, neural_pad, sentence_pad, day_idxs


def add_constant_offset(rng, inputs, std=0.2):
    B, _, D = inputs.shape[0]
    bias = std * jax.random.normal(rng, (B, D))
    return inputs + bias[:, None, :]

def add_gaussian_noise(rng, inputs, std=0.8):
    return inputs + std * jax.random.normal(rng, inputs.shape)

# def gaussian_smooth():
#     mean = (size - 1) / 2
#     kernel = 1.0 / (sigma * jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(- ((jnp.arange(size)-mean)**2)/2)
#     kernel = kernel / jnp.sum(kernel)
#     batch_conv = jax.vmap(lambda x : jsp.signal.convolve(x, kernel, mode='same'))
#     d_batch_conv = jax.vmap(lambda x : batch_conv(x.T).T)
#     out2 = d_batch_conv(xin)
#     return

def train_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm, lr_params):
    """
    Training function for an epoch that loops over batches.
    """
    # Store Metrics
    model = model(training=True)
    batch_losses = []

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs, labels, integration_times, neural_pad, sentence_pad, day_idxs = prep_batch(batch, seq_len, in_dim)
        rng, gauss_rng = jax.random.split(rng)
        inputs = add_gaussian_noise(gauss_rng, inputs, std=0.8)
        rng, co_rng = jax.random.split(rng)
        inputs = add_constant_offset(co_rng, inputs, std=0.2)
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            integration_times,
            neural_pad,
            sentence_pad,
            day_idxs,
            model,
            batchnorm,
        )
        batch_losses.append(loss)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step


def validate(state, model, testloader, seq_len, in_dim, batchnorm, step_rescale=1.0):
    """Validation function that loops over batches"""
    model = model(training=False, step_rescale=step_rescale)
    losses, edit_distance, length, preds = np.array([]), np.array([]), np.array([]), np.array([])
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps, neural_pad, sentence_pad, day_idxs = prep_batch(batch, seq_len, in_dim)
        loss, pred = \
            eval_step(inputs, labels, integration_timesteps, state, model, batchnorm, neural_pad, sentence_pad, day_idxs)
        losses = np.append(losses, loss)
        acc = np.array([compute_ctc_accuracy(_logit, _label, _neural_padding, _label_padding) 
            for (_logit, _label, _neural_padding, _label_padding) in 
            zip(pred, labels, neural_pad, sentence_pad)])
        edit_distance = np.append(edit_distance, np.sum(acc, axis=0)[0])
        length = np.append(length, np.sum(acc, axis=0)[1])
    aveloss = np.mean(losses)
    aveaccu = 1 - ((np.sum(edit_distance)) / (np.sum(length)))
    return aveloss, aveaccu


@partial(jax.jit, static_argnums=(8, 9))
def train_step(state,
               rng,
               batch_inputs,
               batch_labels,
               batch_integration_timesteps,
               batch_neural_pad,
               batch_sentence_pad,
               batch_day_idxs,
               model,
               batchnorm,
               ):

    # downsample
    # batch_neural_pad = batch_neural_pad[:, ::4]

    """Performs a single training step given a batch of data"""
    def loss_fn(params):
        if batchnorm:
            logits, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch_inputs, batch_integration_timesteps, batch_day_idxs,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs, batch_integration_timesteps, batch_day_idxs,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        # downsample
        # logits = logits[:, ::4, :]

        loss = np.mean(ctc_loss(logits, batch_neural_pad, batch_labels, batch_sentence_pad))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(4, 5))
def eval_step(batch_inputs,
              batch_labels,
              batch_integration_timesteps,
              state,
              model,
              batchnorm,
              batch_neural_pad,
              batch_sentence_pad,
              batch_day_idxs
              ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs, batch_integration_timesteps, batch_day_idxs,
                             )
    else:
        logits = model.apply({"params": state.params},
                             batch_inputs, batch_integration_timesteps, batch_day_idxs,
                             )

    losses = np.mean(ctc_loss(logits, batch_neural_pad, batch_labels, batch_sentence_pad))

    return losses, logits
