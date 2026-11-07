# Pueue 命令行帮助参考

版本：**Pueue 4.0.4**。采集时间：2026-11-01。

每个命令仅保留一份原始帮助输出，优先使用 `--help`；省略与父命令帮助重复的 `help` 分发器输出。各命令的 `-h, --help` 均用于显示帮助，其重复选项说明统一省略。

## `pueue`

```text
Interact with the Pueue daemon

Use the `--help` long form to get detailed help output on each subcommand!

Usage: pueue [OPTIONS] [COMMAND]

Commands:
  add          Enqueue a task for execution
  remove       Remove tasks from the list. Running or paused tasks need to be killed first
  switch       Switches the queue position of two commands
  stash        Stash a task. Stashed tasks won't be automatically started
  enqueue      Enqueue stashed tasks. They'll be handled normally afterwards
  start        Resume operation of specific tasks or groups of tasks
  restart      Restart failed or successful task(s)
  pause        Either pause running tasks or specific groups of tasks
  kill         Kill specific running tasks or whole task groups
  send         Send something to a task. Useful for sending confirmations such as 'y\n'
  edit         Adjust editable properties of a task
  env          Use this to add or remove environment variables from tasks
  group        Use this to add or remove groups
  status       Display the current status of all tasks
  log          Display the log output of finished tasks
  follow       Follow the output of a currently running task. This command works like "tail -f"
  wait         Wait until tasks are finished
  clean        Remove all finished tasks from the list
  reset        Kill all tasks, clean up afterwards and reset EVERYTHING!
  shutdown     Remotely shut down the daemon. Should only be used if the daemon isn't started by a
               service manager
  parallel     Set the amount of allowed parallel tasks
  completions  Generates shell completion files
  help         Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...         Verbose mode (-v, -vv, -vvv)
      --color <COLOR>      Colorize the output; auto enables color output when connected to a tty
                           [default: auto] [possible values: auto, never, always]
  -c, --config <CONFIG>    If provided, Pueue only uses this config file
  -p, --profile <PROFILE>  The name of the profile that should be loaded from your config file
  -V, --version            Print version
```

### `pueue add`

```text
Enqueue a task for execution.

There're many different options when scheduling a task. Check the individual option help texts for
more information. Furthermore, please remember that scheduled commands are executed via your system
shell. This means that the command needs proper shell escaping. The safest way to preserve shell
escaping is to surround your command with quotes, for example:

pueue add 'ls $HOME && echo \"Some string\"'

Usage: pueue add [OPTIONS] <COMMAND>...

Arguments:
  <COMMAND>...
          The command to be added

Options:
  -w, --working-directory <working-directory>
          Specify current working directory

  -e, --escape
          Escape any special shell characters (" ", "&", "!", etc.).
          Beware: This implicitly disables nearly all shell specific syntax ("&&", "&>").

  -i, --immediate
          Immediately start the task

      --follow
          Immediately follow a task, if it's started with --immediate

  -s, --stashed
          Create the task in Stashed state.
          
          Useful to avoid immediate execution if the queue is empty.

  -d, --delay <delay>
          Prevents the task from being enqueued until 'delay' elapses. See "enqueue" for accepted
          formats

  -g, --group <GROUP>
          Assign the task to a group.
          
          Groups kind of act as separate queues. All groups run in parallel and you can specify the
          amount of parallel tasks for each group. If no group is specified, the default group will
          be used.
          
          Create groups via the `pueue group` subcommand

  -a, --after <after>...
          Start the task once all specified tasks have successfully finished.
          
          As soon as one of the dependencies fails, this task will fail as well.

  -o, --priority <PRIORITY>
          Start this task with a higher priority.
          
          The higher the number, the faster it will be processed.

  -l, --label <LABEL>
          Add some information for yourself.
          
          This string will be shown in the "status" table. There's no additional logic connected to
          it.

  -p, --print-task-id
          Only return the task id instead of a text.
          
          This is useful when working with dependencies in scripts.
```

### `pueue remove`

```text
Remove tasks from the list. Running or paused tasks need to be killed first

Usage: pueue remove <TASK_IDS>...

Arguments:
  <TASK_IDS>...  The task ids to be removed

```

### `pueue switch`

```text
Switches the queue position of two commands.

Only works on queued and stashed commands.

Usage: pueue switch <TASK_ID_1> <TASK_ID_2>

Arguments:
  <TASK_ID_1>
          The first task id

  <TASK_ID_2>
          The second task id

```

### `pueue stash`

```text
Stash a task. Stashed tasks won't be automatically started.

The enqueue an item, use the `pueue enqueue` subcommand. Stashed entries can also be explicitely
started via `pueue start $task_id`.

Usage: pueue stash [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          Stash these specific tasks

Options:
  -g, --group <GROUP>
          Stash all queued tasks in a group

  -a, --all
          Stash all queued tasks across all groups

  -d, --delay <delay>
          Delay enqueuing these tasks until 'delay' elapses. See DELAY FORMAT below
```

### `pueue enqueue`

```text
Enqueue stashed tasks. They'll be handled normally afterwards.

Enqueues all stashed task in the default group if no arguments are given.

Usage: pueue enqueue [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          Enqueue these specific tasks

Options:
  -g, --group <GROUP>
          Enqueue all stashed tasks in a group

  -a, --all
          Enqueue all stashed tasks across all groups

  -d, --delay <delay>
          Delay enqueuing these tasks until 'delay' elapses. See DELAY FORMAT below


DELAY FORMAT:

    The --delay argument must be either a number of seconds or a "date expression" similar to GNU
    "date -d" with some extensions. It does not attempt to parse all natural language, but is
    incredibly flexible. Here are some supported examples.

    2020-04-01T18:30:00   // RFC 3339 timestamp
    2020-4-1 18:2:30      // Optional leading zeros
    2020-4-1 5:30pm       // Informal am/pm time
    2020-4-1 5pm          // Optional minutes and seconds
    April 1 2020 18:30:00 // English months
    1 Apr 8:30pm          // Implies current year
    4/1                   // American form date
    wednesday 10:30pm     // The closest wednesday in the future at 22:30
    wednesday             // The closest wednesday in the future
    4 months              // 4 months from today at 00:00:00
    1 week                // 1 week at the current time
    1days                 // 1 day from today at the current time
    1d 03:00              // The closest 3:00 after 1 day (24 hours)
    3h                    // 3 hours from now
    3600s                 // 3600 seconds from now
```

### `pueue start`

```text
Resume operation of specific tasks or groups of tasks.

Without any parameters this resumes the default group and all its tasks. Can also be used
force-start specific tasks, which **ignores** any group parallelism limits or dependencies a task my
have!

Usage: pueue start [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          Start these specific tasks. Paused tasks will resumed. Queued/Stashed tasks will be
          force-started

Options:
  -g, --group <GROUP>
          Resume a specific group and all paused tasks in it.
          
          The group will be set to running and its paused tasks will be resumed.

  -a, --all
          Resume all groups!
          
          All groups will be set to running and paused tasks will be resumed.
```

### `pueue restart`

```text
Restart failed or successful task(s).

By default, identical tasks will be created and enqueued, but it's possible to restart in-place. You
can also edit a few properties, such as the path and the command, before restarting.

Usage: pueue restart [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          Restart these specific tasks

Options:
  -a, --all-failed
          Restart all failed tasks across all groups.
          
          This is nice for usage in combination with `-i/--in-place` (or the respective config
          option).

  -g, --failed-in-group <FAILED_IN_GROUP>
          Like `--all-failed`, but only restart tasks failed tasks of a specific group

  -k, --immediate
          Immediately start the tasks, no matter how many open slots there are. This will ignore any
          dependencies tasks may have

  -s, --stashed
          Set the restarted task to a "Stashed" state. Useful to avoid immediate execution

  -i, --in-place
          Restart the task by reusing the already existing tasks. This will overwrite any previous
          logs of the restarted tasks.
          
          This can also be enabled by default via the `restart_in_place` config option.

      --not-in-place
          Restart the task by creating a new identical tasks. Only necessary if you have the
          `restart_in_place` configuration set to true

  -e, --edit
          Edit the task before restarting
```

### `pueue pause`

```text
Either pause running tasks or specific groups of tasks.

By default, pauses the default group and all its tasks. A paused queue (group) won't start any new
tasks.

Usage: pueue pause [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          Pause these specific tasks

Options:
  -g, --group <GROUP>
          Pause a specific group

  -a, --all
          Pause all groups!

  -w, --wait
          Pause the specified groups, but let already running tasks finish by themselves
```

### `pueue kill`

```text
Kill specific running tasks or whole task groups.

Kills all tasks of the default group when no ids or a specific group are provided.

Usage: pueue kill [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          Kill these specific tasks

Options:
  -g, --group <GROUP>
          Kill all running tasks in a group. This also pauses the group

  -a, --all
          Kill all running tasks across ALL groups. This also pauses all groups

  -s, --signal <SIGNAL>
          Send a UNIX signal instead of simply killing the process.
          
          DISCLAIMER: This bypasses Pueue's process handling logic! You might enter weird invalid
          states, use at your own descretion.
          
          This argument also excepts the integer representation as well as the signal short name.
          E.g. `sigint`, `int`, or `2` are the same.
```

### `pueue send`

```text
Send something to a task. Useful for sending confirmations such as 'y\n'

Usage: pueue send <TASK_ID> <INPUT>

Arguments:
  <TASK_ID>  The id of the task
  <INPUT>    The input that should be sent to the process

```

### `pueue edit`

```text
Adjust editable properties of a task.

Only stashed or queued tasks can be edited. A temporary folder folder/file will be opened by your
$EDITOR to edit the tasks.

Usage: pueue edit [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          The ids of all tasks that should be edited

```

### `pueue env`

```text
Use this to add or remove environment variables from tasks

Usage: pueue env <COMMAND>

Commands:
  set    Set a variable for a specific task's environment
  unset  Remove a specific variable from a task's environment
  help   Print this message or the help of the given subcommand(s)

```

#### `pueue env set`

```text
Set a variable for a specific task's environment

Usage: pueue env set <TASK_ID> <KEY> <VALUE>

Arguments:
  <TASK_ID>  The id of the task for which the variable should be set
  <KEY>      The name of the environment variable to set
  <VALUE>    The value of the environment variable to set

```

#### `pueue env unset`

```text
Remove a specific variable from a task's environment

Usage: pueue env unset <TASK_ID> <KEY>

Arguments:
  <TASK_ID>  The id of the task for which the variable should be set
  <KEY>      The name of the environment variable to set

```

### `pueue group`

```text
Use this to add or remove groups.

By default, this will simply display all known groups.

Usage: pueue group [OPTIONS] [COMMAND]

Commands:
  add     Add a group by name
  remove  Remove a group by name. This will move all tasks in this group to the default group!
  help    Print this message or the help of the given subcommand(s)

Options:
  -j, --json
          Print the list of groups as json
```

#### `pueue group add`

```text
Add a group by name

Usage: pueue group add [OPTIONS] <NAME>

Arguments:
  <NAME>
          

Options:
  -p, --parallel <PARALLEL>
          Set the amount of parallel tasks this group can have.
          
          Setting this to 0 means an unlimited amount of parallel tasks.
```

#### `pueue group remove`

```text
Remove a group by name. This will move all tasks in this group to the default group!

Usage: pueue group remove <NAME>

Arguments:
  <NAME>  

```

### `pueue status`

```text
Display the current status of all tasks

Usage: pueue status [OPTIONS] [QUERY]...

Arguments:
  [QUERY]...
          Users can specify a custom query to filter for specific values, order by a column
          or limit the amount of tasks listed.
          
          Syntax:
             [column_selection]? [filter]* [order_by]? [limit]?
          
          where:
            - column_selection := `columns=[column]([column],)*`
            - column := `id | status | command | label | path | enqueue_at | dependencies | start |
            end`
            - filter := `[filter_column] [filter_op] [filter_value]`
              (note: not all columns support all operators, see "Filter columns" below.)
            - filter_column := `status | command | label | start | end | enqueue_at`
            - filter_op := `= | != | < | > | %=`
              (`%=` means 'contains', as in the test value is a substring of the column value)
            - order_by := `order_by [column] [order_direction]`
            - order_direction := `asc | desc`
            - limit := `[limit_type]? [limit_count]`
            - limit_type := `first | last`
            - limit_count := a positive integer
          
          Filter columns:
            - `status` supports the operators `=`, `!=`
              against test values that are:
                - strings like `queued`, `stashed`, `paused`, `running`, `success`, `failed`
            - `command`, `label` support the operators `=`, `!=`, `%=`
              against test values that are:
                - strings like `some text`
            - `start`, `end`, `enqueue_at` contain a datetime
              which support the operators `=`, `!=`, `<`, `>`
              against test values that are:
                - date like `YYYY-MM-DD`
                - time like `HH:mm:ss` or `HH:mm`
                - datetime like `YYYY-MM-DDHH:mm:ss`
                  (note there is currently no separator between the date and the time)
          
          Examples:
            - `status=running`
            - `command%=echo`
            - `label=mytask`
            - `columns=id,status,command status=running start > 2023-05-2112:03:17 order_by command
            first 5`
          
          The formal syntax is defined here:
          https://github.com/Nukesor/pueue/blob/main/pueue/src/client/commands/state/query/syntax.pest
          
          More documentation is on the query syntax PR:
          https://github.com/Nukesor/pueue/issues/350#issue-1359083118

Options:
  -j, --json
          Print the current state as json to stdout. This does not include the output of tasks. Use
          `log -j` if you want everything

  -g, --group <GROUP>
          Only show tasks of a specific group
```

### `pueue log`

```text
Display the log output of finished tasks.

Only the last few lines will be shown by default. If you want to follow the output of a task, please
use the \"follow\" subcommand.

Usage: pueue log [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          View the task output of these specific tasks

Options:
  -g, --group <GROUP>
          View the outputs of this specific group's tasks

  -a, --all
          Show the logs of all groups' tasks

  -j, --json
          Print the resulting tasks and output as json.
          
          By default only the last lines will be returned unless --full is provided. Take care, as
          the json cannot be streamed! If your logs are really huge, using --full can use all of
          your machine's RAM.

  -l, --lines <LINES>
          Only print the last X lines of each task's output.
          
          This is done by default if you're looking at multiple tasks.

  -f, --full
          Show the whole output
```

### `pueue follow`

```text
Follow the output of a currently running task. This command works like "tail -f"

Usage: pueue follow [OPTIONS] [TASK_ID]

Arguments:
  [TASK_ID]
          The id of the task you want to watch.
          
          If no or multiple tasks are running, you have to specify the id. If only a single task is
          running, you can omit the id.

Options:
  -l, --lines <LINES>
          Only print the last X lines of the output before following
```

### `pueue wait`

```text
Wait until tasks are finished.

By default, this will wait for all tasks in the default group to finish.

Note: This will also wait for all tasks that aren't somehow 'Done'. Includes: [Paused, Stashed,
Locked, Queued, ...]

Usage: pueue wait [OPTIONS] [TASK_IDS]...

Arguments:
  [TASK_IDS]...
          This allows you to wait for specific tasks to finish

Options:
  -g, --group <GROUP>
          Wait for all tasks in a specific group

  -a, --all
          Wait for all tasks across all groups and the default group

  -q, --quiet
          Don't show any log output while waiting

  -s, --status <STATUS>
          Wait for tasks to reach a specific task status
```

### `pueue clean`

```text
Remove all finished tasks from the list

Usage: pueue clean [OPTIONS]

Options:
  -s, --successful-only  Only clean tasks that finished successfully
  -g, --group <GROUP>    Only clean tasks of a specific group
```

### `pueue reset`

```text
Kill all tasks, clean up afterwards and reset EVERYTHING!

Usage: pueue reset [OPTIONS]

Options:
  -g, --groups <GROUPS>  If groups are specified, only those specific groups will be reset
  -f, --force            Don't ask for any confirmation
```

### `pueue shutdown`

```text
Remotely shut down the daemon. Should only be used if the daemon isn't started by a service manager

Usage: pueue shutdown

```

### `pueue parallel`

```text
Set the amount of allowed parallel tasks

By default, adjusts the amount of the default group.

No tasks will be stopped, if this is lowered. This limit is only considered when tasks are
scheduled.

Usage: pueue parallel [OPTIONS] [PARALLEL_TASKS]

Arguments:
  [PARALLEL_TASKS]
          The amount of allowed parallel tasks.
          
          Setting this to 0 means an unlimited amount of parallel tasks.

Options:
  -g, --group <group>
          Set the amount for a specific group
```

### `pueue completions`

```text
Generates shell completion files.

This can be ignored during normal operations.

Usage: pueue completions <SHELL> [OUTPUT_DIRECTORY]

Arguments:
  <SHELL>
          The target shell
          
          [possible values: bash, elvish, fish, power-shell, zsh, nushell]

  [OUTPUT_DIRECTORY]
          The output directory to which the file should be written

```
