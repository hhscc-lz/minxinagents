FROM langchain/langgraph-api:3.11

# -- Install Chinese fonts for matplotlib chart rendering --
# (placed before COPY so this layer is cached independently of code changes)
RUN apt-get update && apt-get install -y --no-install-recommends fonts-wqy-zenhei && rm -rf /var/lib/apt/lists/*

# -- Configure matplotlib: default font + headless backend --
RUN mkdir -p /root/.config/matplotlib
COPY libs/cli/matplotlibrc /root/.config/matplotlib/matplotlibrc

# -- Adding local packages from monorepo --
COPY libs/cli /deps/cli
COPY libs/deepagents /deps/deepagents
# -- End of local package copy --
COPY libs/cli/cert /deps/cli/cert

RUN mkdir -p /root/.deepagents
# COPY libs/cli/config-glm.toml /root/.deepagents/config.toml
COPY libs/cli/config.toml /root/.deepagents/config.toml

# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGSERVE_GRAPHS='{"deepagent": "/deps/cli/deepagents_cli/server_graph.py:graph"}'



# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

RUN mkdir -p /minxinagent

WORKDIR /minxinagent

EXPOSE 8000

ENTRYPOINT ["python", "/deps/cli/textual_run.py"]
