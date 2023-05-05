import matplotlib.pyplot as plt
import optax

from src.training.optim import create_learning_rate_fn


def test_create_learning_rate_fn():
    num_epochs = 100
    steps_per_epoch = 100
    base_lr = 1e-3
    warmup_epochs = 10
    cosine_decay_epochs = 50

    learning_rate_fn = create_learning_rate_fn(
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        cosine_decay_epochs=cosine_decay_epochs
    )

    learning_rates = []
    for step in range(num_epochs * steps_per_epoch):
        lr = learning_rate_fn(step)
        # print(f"step: {step}, lr: {learning_rate_fn(step)}")
        learning_rates.append(lr)

    plt.plot(learning_rates)
    plt.show()


def test_cosine_decay_schedule():
    num_steps = 1000

    # cosine_decay_schedule_fn = optax.cosine_decay_schedule(
    #     init_value=1.0,
    #     decay_steps=num_steps,
    #     alpha=0.0,
    #     exponent=1.0,
    # )
    # schedule = lambda _step: 1.0 - cosine_decay_schedule_fn(_step)
    schedule = lambda _step: 1.0 - optax.cosine_decay_schedule(
        init_value=1.0,
        decay_steps=num_steps,
        alpha=0.0,
        exponent=1.0,
    )(_step)

    learning_rates = []
    for step in range(num_steps):
        lr = schedule(step)
        # print(f"step: {step}, lr: {learning_rate_fn(step)}")
        learning_rates.append(lr)

    plt.plot(learning_rates)
    plt.show()


if __name__ == "__main__":
    # test_create_learning_rate_fn()
    test_cosine_decay_schedule()
