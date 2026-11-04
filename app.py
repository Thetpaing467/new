    ok = set(inspect.signature(demo.launch).parameters)
    lk = {k: v for k, v in launch_kw.items() if k in ok}
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT, show_error=True, **lk)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and not hasattr(args, "height"):
        args.height = 720
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit("\n⛔ ရပ်လိုက်ပါပြီ (ပြန် run ရင် ပြီးသား အသံတွေ cache ကနေ ယူမယ်)")
    except Exception as ex:
        sys.exit(f"\n❌ {ex}")


if __name__ == "__main__":
    # no arguments -> web UI (same behaviour as before); otherwise CLI
    if len(sys.argv) == 1:
        launch_ui()
    else:
        main()
