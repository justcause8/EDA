from io import BytesIO
import base64
import matplotlib.pyplot as plt


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def html_block(title, content):
    return f"""
    <section>
        <h2>{title}</h2>
        {content}
    </section>
    """
